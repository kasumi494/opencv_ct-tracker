#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/core/core.hpp>

typedef std::vector <cv::Rect> ListOfRects;

struct Feature
{
    cv::Rect rect;
    float weight;
};

class Tracker {
public:
    Tracker ();
    ~Tracker () {}

    void Init (cv::Mat& startFrame,  cv::Rect const &target);

    cv::Rect TrackObject (const cv::Mat& frame);
    void DrawObject (cv::Mat &where) const;

private:
    void ComputeFilterBank ();

    void GetSampleRect (float max_radius, float min_radius, int max_num_samples, ListOfRects &samples);
    void GetSampleRect (float max_radius, ListOfRects &samples);
    void GetFeatureValue (ListOfRects &samples, cv::Mat &low_dim_features);

    void UpdateClassifier (cv::Mat &low_dim_features, std::vector<float> &mu, std::vector<float> &sigma);
    int RadioClassifier (cv::Mat &low_dim_features);

    int min_num_rects_;
    int max_num_rects_;
    int num_features_;
    int max_radius_;
    int search_window_;
    float learn_rate_;

    cv::Size frame_size_;
    std::vector <std::vector <Feature> > filters_;
    ListOfRects positive_sample_box_;
    ListOfRects negative_sample_box_;

    cv::Mat_ <float> integral_image_;

    cv::Mat positive_feature_value_;
    cv::Mat negative_feature_value_;

    std::vector<float> positive_mu_;
    std::vector<float> positive_sigma_;
    std::vector<float> negative_mu_;
    std::vector<float> negative_sigma_;

    ListOfRects detect_boxes_;
    cv::Mat detect_feature_value_;
    cv::RNG rng;

    cv::Rect objectPosition_;
};

#endif // TRACKER_H
