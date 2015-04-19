#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#ifdef ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include "neural.h"

void 
neural_feed_forward(struct neural_net_layer *head, float *x) 
{
	/* point "current" layer to head and set input as feature array x */
	struct neural_net_layer *curr = head;
	struct neural_net_layer *next;
	curr->act = x;

	int rows;
	int cols;

	while ((next = curr->next) != NULL) {
		rows = curr->out_nodes;
		cols = curr->in_nodes;

		/* act = theta * x + act */
		memcpy(next->act, curr->bias, rows * sizeof(float));
		cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, curr->theta, 
			cols, curr->act, 1, 1.0, next->act, 1);

		/* apply sigmoid function */
		for (int i = 0; i < rows; i++) {
			next->act[i] = 1.0 / (1 + expf(-next->act[i]));
		}

		/* transition to next layer */
		curr = next;
	}
}

void 
neural_back_prop(struct neural_net_layer *tail, float *y, const float alpha, 
		const float lambda) 
{
	/* get delta from final predictions vs. actual; delta = a - y */
	cblas_scopy(tail->in_nodes, tail->act, 1, tail->delta, 1);
	cblas_saxpy(tail->in_nodes, -1.0, y, 1, tail->delta, 1);

	struct neural_net_layer *curr;
	struct neural_net_layer *prev;
	int rows;
	int cols;

	for (curr = tail; curr->prev != NULL; curr = prev) {
		prev = curr->prev;
		rows = prev->out_nodes;
		cols = prev->in_nodes;

		/* delta^(n-1) = theta^(n-1)T * delta^(n) .* a^(n-1) (1 - a^(n-1)) */
		cblas_sgemv(CblasRowMajor, CblasTrans, rows, cols, 1.0, prev->theta, 
				cols, curr->delta, 1, 0.0, prev->delta, 1); 
		for (int j = 0; j < cols; j++) {
			prev->delta[j] *= (prev->act[j] * (1 - prev->act[j]));
		}
	}

	for (curr = tail; curr->prev != NULL; curr = prev) {
		prev = curr->prev;
		rows = prev->out_nodes;
		cols = prev->in_nodes;

		/* bias -= alpha * delta */
		cblas_saxpy(rows, -alpha, curr->delta, 1, prev->bias, 1);

		/* theta -= alpha * (delta * actT + lambda * theta) */
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, 1, -alpha, 
				curr->delta, 1, prev->act, cols, 1 - alpha * lambda, prev->theta, cols);
	}
}

void
neural_sgd_iteration(struct neural_net_layer *head, struct neural_net_layer *tail, 
		float *features, float *labels, const int n_samples, 
		const float alpha, const float lambda)
{
	int n_features = head->in_nodes;
	int n_labels = tail->in_nodes;

	for (int i = 0; i < n_samples; i++) {
		neural_feed_forward(head, features + i * n_features);
		neural_back_prop(tail, labels + i * n_labels, alpha, lambda);
	}
}

void
neural_predict_prob(struct neural_net_layer *head, struct neural_net_layer *tail,
		float *features, float *preds, const int n_samples)
{
	int n_features = head->in_nodes;
	int n_labels = tail->in_nodes;

	for (int i = 0; i < n_samples; i++) {
		neural_feed_forward(head, features + i * n_features);
		memcpy(preds + i * n_labels, tail->act, n_labels * sizeof(float));
	}
}
