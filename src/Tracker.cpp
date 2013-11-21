#include "Tracker.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Tracker::Tracker ()
{
    min_num_rects_ = 2;
    max_num_rects_ = 10;
    num_features_ = 50;

    max_radius_ = 4;
    search_window_ = 25;

    positive_mu_ = std::vector <float> (num_features_, 0.0f);
    negative_mu_ = std::vector <float> (num_features_, 0.0f);
    positive_sigma_ = std::vector <float> (num_features_, 1.0f);
    negative_sigma_ = std::vector <float> (num_features_, 1.0f);

    learn_rate_ = 0.85f;  // Learning rate parameter
}

void Tracker::ComputeFilterBank ()
{
    filters_.resize (num_features_);

    Feature temp;
    for (int i = 0; i < num_features_; ++i)
    {
        int num_rect = cvFloor (rng.uniform (static_cast<double> (min_num_rects_),
                                             static_cast<double> (max_num_rects_)));

        for (int j = 0; j < num_rect; ++j)
        {
            temp.rect.x = cvFloor (rng.uniform (0.0, objectPosition_.width - 3.0));
            temp.rect.y = cvFloor (rng.uniform (0.0, objectPosition_.height - 3.0));
            temp.rect.width = cvCeil (rng.uniform (0.0, objectPosition_.width - temp.rect.x - 2.0));
            temp.rect.height = cvCeil (rng.uniform (0.0, objectPosition_.height - temp.rect.y - 2.0));

            temp.weight = pow (-1.0, cvFloor (rng.uniform (0.0, 3.0))) / sqrtf(num_rect);

            filters_[i].push_back(temp);
        }
    }
}

void Tracker::GetSampleRect (float max_radius, float min_radius, int max_num_samples, ListOfRects &samples)
{
    samples.clear();

    cv::Point2i min_border, max_border;
    min_border.y = max (0, objectPosition_.y - static_cast<int> (max_radius));
    min_border.x = max (0, objectPosition_.x - static_cast<int> (max_radius));

    max_border.x = min (frame_size_.width - objectPosition_.width,
                        objectPosition_.x + static_cast<int> (max_radius));
    max_border.y = min (frame_size_.height - objectPosition_.height,
                        objectPosition_.y + static_cast<int> (max_radius));

    float prob = max_num_samples /
                 ((max_border.y - min_border.y + 1.0f) * (max_border.x - min_border.x + 1.0f));

    float max_radius2 = max_radius * max_radius;
    float min_radius2 = min_radius * min_radius;

    for (int row = min_border.y; row <= max_border.y; ++row)
    {
        for (int col = min_border.x; col <= max_border.x; ++col)
        {
            int dist = (objectPosition_.y - row) * (objectPosition_.y - row) +
                       (objectPosition_.x - col) * (objectPosition_.x - col);

            if (rng.uniform (0.0, 1.0) < prob && dist < max_radius2 && dist >= min_radius2)
                samples.push_back (cv::Rect (col, row, objectPosition_.width, objectPosition_.height));
        }
    }
}

void Tracker::GetSampleRect (float max_radius, ListOfRects &samples)
{
    samples.clear();

    cv::Point2i min_border, max_border;
    min_border.y = max (0, objectPosition_.y - static_cast<int> (max_radius));
    min_border.x = max (0, objectPosition_.x - static_cast<int> (max_radius));

    max_border.x = min (frame_size_.width - objectPosition_.width,
                        objectPosition_.x + static_cast<int> (max_radius));
    max_border.y = min (frame_size_.height - objectPosition_.height,
                        objectPosition_.y + static_cast<int> (max_radius));

    float max_radius2 = max_radius * max_radius;

    for (int row = min_border.y; row <= max_border.y; ++row)
    {
        for (int col = min_border.x; col <= max_border.x; ++col)
        {
            int dist = (objectPosition_.y - row) * (objectPosition_.y - row) +
                       (objectPosition_.x - col) * (objectPosition_.x - col);

            if (dist < max_radius2)
                samples.push_back (cv::Rect(col, row, objectPosition_.width, objectPosition_.height));
        }
    }
}

void Tracker::GetFeatureValue (ListOfRects &samples, cv::Mat &low_dim_features)
{
    low_dim_features = Mat::zeros (num_features_, samples.size(), CV_32F);

    for (int i = 0, lim_i = filters_.size(); i < lim_i; ++i)
    {
        for (int j = 0, lim_j = samples.size(); j < lim_j; ++j)
        {
            for (int k = 0, lim_k = filters_[i].size(); k < lim_k; ++k)
            {
                int min_x = samples[j].x + filters_[i][k].rect.x;
                int max_x = min_x + filters_[i][k].rect.width;

                int min_y = samples[j].y + filters_[i][k].rect.y;
                int max_y = min_y + filters_[i][k].rect.height;

                low_dim_features.at<float> (i, j) += filters_[i][k].weight *
                            (   integral_image_(min_y, min_x) +
                                integral_image_(max_y, max_x) -
                                integral_image_(min_y, max_x) -
                                integral_image_(max_y, min_x) );
            }
        }
    }
}

void Tracker::UpdateClassifier (cv::Mat &low_dim_features, std::vector<float> &mu, std::vector<float> &sigma)
{
    for (int i = 0; i < low_dim_features.rows; ++i)
    {
        cv::Scalar _mu, _sigma;

        meanStdDev (low_dim_features.row(i), _mu, _sigma);

        // equation 6 in paper
        sigma[i] = sqrtf (learn_rate_ * sigma[i] * sigma[i] +
                            (1.0f - learn_rate_) * _sigma.val[0] * _sigma.val[0] +
                            learn_rate_ * (1.0f - learn_rate_) *
                            (mu[i] - _mu.val[0]) * (mu[i] - _mu.val[0]));

        mu[i] = mu[i] * learn_rate_ + (1.0f - learn_rate_) * _mu.val[0];
    }
}

int Tracker::RadioClassifier (cv::Mat &low_dim_features)
{
    // equation 4

    float radio_max = -FLT_MAX;
    int radio_max_index = 0;

    for (int j = 0; j < low_dim_features.cols; ++j)
    {
        float sumRadio = 0.0f;
        for (int i = 0; i < low_dim_features.rows; ++i)
        {
            float pos_p = exp ((low_dim_features.at<float> (i, j) - positive_mu_[i]) *
                               (low_dim_features.at<float> (i, j) - positive_mu_[i]) /
                               -(2.0f * positive_sigma_[i] * positive_sigma_[i] + 1e-30)) /
                                (positive_sigma_[i] + 1e-30);

            float neg_p = exp ((low_dim_features.at<float> (i, j) - negative_mu_[i]) *
                               (low_dim_features.at<float> (i, j) - negative_mu_[i]) /
                               -(2.0f * negative_sigma_[i] * negative_sigma_[i] + 1e-30) ) /
                                (negative_sigma_[i] + 1e-30);

            sumRadio += log (pos_p + 1e-30) - log (neg_p + 1e-30);
        }

        if (radio_max < sumRadio)
        {
            radio_max = sumRadio;
            radio_max_index = j;
        }
    }

    return radio_max_index;
}

Rect Tracker::TrackObject (const Mat &frame)
{
    GetSampleRect (search_window_, detect_boxes_);
    integral (frame, integral_image_, CV_32F);
    GetFeatureValue (detect_boxes_, detect_feature_value_);

    int radio_max_index = RadioClassifier (detect_feature_value_);
    objectPosition_ = detect_boxes_[radio_max_index];

    // update
    GetSampleRect (max_radius_, 0.0, 1000000, positive_sample_box_);
    GetSampleRect (search_window_ * 1.5, max_radius_ + 10.0, 100, negative_sample_box_);

    GetFeatureValue (positive_sample_box_, positive_feature_value_);
    GetFeatureValue (negative_sample_box_, negative_feature_value_);
    UpdateClassifier (positive_feature_value_, positive_mu_, positive_sigma_);
    UpdateClassifier (negative_feature_value_, negative_mu_, negative_sigma_);

    return objectPosition_;
}

void Tracker::Init (Mat &startFrame, const Rect &target)
{
    objectPosition_ = target;
    frame_size_ = startFrame.size();

    ComputeFilterBank ();

    GetSampleRect (max_radius_, 0, 1000000, positive_sample_box_);
    GetSampleRect (search_window_ * 1.5, max_radius_ + 4.0, 100, negative_sample_box_);

    integral (startFrame, integral_image_, CV_32F);

    GetFeatureValue (positive_sample_box_, positive_feature_value_);
    GetFeatureValue (negative_sample_box_, negative_feature_value_);
    UpdateClassifier (positive_feature_value_, positive_mu_, positive_sigma_);
    UpdateClassifier (negative_feature_value_, negative_mu_, negative_sigma_);
}


void Tracker::DrawObject (cv::Mat &where) const
{
    rectangle (where, objectPosition_, cv::Scalar(0, 0, 255), 3);
}
