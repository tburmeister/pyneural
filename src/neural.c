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
neural_feed_forward(struct neural_net_layer *head, float *x, int batch_size) 
{
	/* point "current" layer to head and set input as feature array x */
	struct neural_net_layer *curr = head;
	struct neural_net_layer *next;
	curr->act = x;

	int in_nodes;
	int out_nodes;

	while ((next = curr->next) != NULL) {
		in_nodes = curr->in_nodes;
		out_nodes = curr->out_nodes;

		/* act = X * Theta^T + bias */
		for (int i = 0; i < batch_size; i++) {
			memcpy(next->act + i * out_nodes, curr->bias, out_nodes * sizeof(float));
		}
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, out_nodes, in_nodes,
			1.0, curr->act, in_nodes, curr->theta, in_nodes, 1.0, next->act, out_nodes);

		/* apply sigmoid function */
		for (int i = 0; i < out_nodes * batch_size; i++) {
			next->act[i] = 1.0 / (1 + expf(-next->act[i]));
		}

		/* transition to next layer */
		curr = next;
	}
}

void 
neural_sgd_feed_forward(struct neural_net_layer *head, float *x) 
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
neural_back_prop(struct neural_net_layer *tail, float *y, const int batch_size,
		const float alpha, const float lambda) 
{
	/* get delta from final predictions vs. actual; delta = a - y */
	cblas_scopy(tail->in_nodes * batch_size, tail->act, 1, tail->delta, 1);
	cblas_saxpy(tail->in_nodes * batch_size, -1.0, y, 1, tail->delta, 1);

	struct neural_net_layer *curr;
	struct neural_net_layer *prev;
	int in_nodes;
	int out_nodes;

	for (curr = tail; curr->prev != NULL; curr = prev) {
		prev = curr->prev;
		in_nodes = prev->in_nodes;
		out_nodes = prev->out_nodes;

		/* Delta^(n-1) = Delta^(n) * Theta^(n-1) .* A^(n-1) (1 - A^(n-1)) */
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size, in_nodes, out_nodes,
				1.0, curr->delta, out_nodes, prev->theta, in_nodes, 0.0, prev->delta, in_nodes);
		for (int j = 0; j < in_nodes * batch_size; j++) {
			prev->delta[j] *= (prev->act[j] * (1 - prev->act[j]));
		}
	}

	for (curr = tail; curr->prev != NULL; curr = prev) {
		prev = curr->prev;
		in_nodes = prev->in_nodes;
		out_nodes = prev->out_nodes;

		/* bias -= alpha * delta */
		for (int i = 0; i < batch_size; i++) {
			cblas_saxpy(out_nodes, -alpha, curr->delta + i * out_nodes, 1, prev->bias, 1);
		}

		/* Theta -= alpha * (Delta^T * Act + lambda * Theta) */
		cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_nodes, in_nodes, batch_size,
				-alpha, curr->delta, out_nodes, prev->act, in_nodes, 1 - alpha * lambda, 
				prev->theta, in_nodes);
	}
}

void 
neural_sgd_back_prop(struct neural_net_layer *tail, float *y, const float alpha, 
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
neural_train_iteration(struct neural_net_layer *head, struct neural_net_layer *tail, 
		float *features, float *labels, const int n_samples, const int batch_size, 
		const float alpha, const float lambda)
{
	int n_features = head->in_nodes;
	int n_labels = tail->in_nodes;
	int i;

	if (batch_size == 1) {
		for (i = 0; i < n_samples; i++) {
			neural_sgd_feed_forward(head, features + i * n_features);
			neural_sgd_back_prop(tail, labels + i * n_labels, alpha, lambda);
		}
	} else {
		for (i = 0; i < n_samples / batch_size; i++) {
			neural_feed_forward(head, features + i * n_features * batch_size, batch_size);
			neural_back_prop(tail, labels + i * n_labels * batch_size, batch_size, alpha, lambda);
		}

		/* take care of remaining examples */
		int remainder = n_samples % batch_size;
		if (remainder > 0) {
			neural_feed_forward(head, features + i * n_features * batch_size, remainder);
			neural_back_prop(tail, labels + i * n_labels * batch_size, remainder, alpha, lambda);
		}
	}
}

void
neural_predict_prob(struct neural_net_layer *head, struct neural_net_layer *tail,
		float *features, float *preds, const int n_samples, const int batch_size)
{
	int n_features = head->in_nodes;
	int n_labels = tail->in_nodes;
	int i;

	if (batch_size == 1) {
		for (i = 0; i < n_samples; i++) {
			neural_sgd_feed_forward(head, features + i * n_features);
			memcpy(preds + i * n_labels, tail->act, n_labels * sizeof(float));
		}
	} else {
		for (i = 0; i < n_samples / batch_size; i++) {
			neural_feed_forward(head, features + i * n_features * batch_size, batch_size);
			memcpy(preds + i * n_labels * batch_size, tail->act, 
					n_labels * batch_size * sizeof(float));
		}

		/* take care of remaining examples */
		int remainder = n_samples % batch_size;
		if (remainder > 0) {
			neural_feed_forward(head, features + i * n_features * batch_size, remainder);
			memcpy(preds + i * n_labels * batch_size, tail->act,
					n_labels * remainder * sizeof(float));
		}
	}
}
