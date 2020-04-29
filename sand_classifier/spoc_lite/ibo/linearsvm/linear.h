#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */  
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
	double *init_sol;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* svm_linear_train(const struct problem *prob, const struct parameter *param);
void svm_linear_cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);
void svm_linear_find_parameter_C(const struct problem *prob, const struct parameter *param, int nr_fold, double start_C, double max_C, double *best_C, double *best_rate);

double svm_linear_predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double svm_linear_predict(const struct model *model_, const struct feature_node *x);
double svm_linear_predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int svm_linear_save_model(const char *model_file_name, const struct model *model_);
struct model *svm_linear_load_model(const char *model_file_name);

int svm_linear_get_nr_feature(const struct model *model_);
int svm_linear_get_nr_class(const struct model *model_);
void svm_linear_get_labels(const struct model *model_, int* label);
double svm_linear_get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double svm_linear_get_decfun_bias(const struct model *model_, int label_idx);

void svm_linear_free_model_content(struct model *model_ptr);
void svm_linear_free_and_destroy_model(struct model **model_ptr_ptr);
void svm_linear_destroy_param(struct parameter *param);

const char *svm_linear_check_parameter(const struct problem *prob, const struct parameter *param);
int svm_linear_check_probability_model(const struct model *model);
int svm_linear_check_regression_model(const struct model *model);
void svm_linear_set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

