/*!
 *  @file    classifier_v1.h
 *  @brief   SVM-based sand classifier
 *  @author  Yumi Iwashita <iwashita@jpl.nasa.gov>
 *  @author  Kyohei Otsu <otsu@jpl.nasa.gov>
 *  @date    2017-07-14
 *
 *  This classes use the SVM-based classifiers developed by Yumi Iwashita.
 */

#ifndef _CLASSIFIER_V1_H_
#define _CLASSIFIER_V1_H_

#include "sand_classifier/classifier.h"

#include "ibo.h"
#include "ibo/lbp.h"
#include "ibo/svm.h"
#include "ibo/svm_linear.h"
#include "classification.h"


namespace spoc
{

  /*! SVM-based classifier class */
  class ClassifierSVM : public Classifier
  {
    public:
      ClassifierSVM(ros::NodeHandle nh, ros::NodeHandle pnh);
      ~ClassifierSVM();

    protected:
      /*! 
       * Detect sand regions and probabilities from color image
       * @param src_bgr  input color image
       * @param dst_labl sand label image
       * @param dst_prob sand probability image
       */
      virtual bool classify(const cv::Mat &src_bgr,
                                  cv::Mat &dst_labl,
                                  cv::Mat &dst_prob);

      /*! Params for LBP (Local Binary Pattern) feature extraction */
      ibo::struct_lbp_parameter lbp_parameter;

      /*! SVM object */
      ibo::LIBSVM svm;

      /*! Options to specify feature combination to use */
      struct_feature_options feature_options;
  };


}  // namespace spoc

#endif  // _CLASSIFIER_V1_H_

