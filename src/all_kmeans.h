#pragma once

#include <opencv2/core.hpp>

double kmeans_base(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
);

double student1_kmeans(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
);

double kmeans_uchar(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
);

double kmeans_uchar16d(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
);