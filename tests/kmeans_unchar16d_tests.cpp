#include "all_kmeans.h"

#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <chrono>
#include <string>

#include <gtest/gtest.h>

using namespace cv;
using ClustersCount = int;
using PointsCount = int;
using Attempts = int;
using Perf = bool;
using IsKMeansPlusPlus = bool;

namespace {
    static constexpr int PERF_ITERATIONS = 10;

    struct KMeansUniversalParams {
        ClustersCount clustersCount;
        PointsCount pointsCount;
        Attempts attempts;
        Perf isPerf;  // also test performance?
        IsKMeansPlusPlus isKMeansPlusPlus;
    };

    const char* bool2chars(const bool value) {
        return value ? "true" : "false";
    }

    std::ostream& operator<<(std::ostream& out, const KMeansUniversalParams& params) {
        out << "KMeansUniversalParams{";
        out << "Clusters{" << params.clustersCount;
        out << "}, Points{" << params.pointsCount;
        out << "}, Attempts{" << params.attempts;
        out << "}, Perf{" << bool2chars(params.isPerf);
        out << "}}";
        return out;
    }

    class SyntheticTestKMeansStudent2_u8c16 : public testing::TestWithParam<KMeansUniversalParams> {
    protected:
        void run(const KMeansUniversalParams& params);
    };

    template <typename Type>
    static inline bool checkDiff(const Type* actual, const Type* ref, const int size, float tolerance, const std::string string) {
        for (int i = 0; i < size; i++) {
            float diff = std::abs(actual[i] - ref[i]);
            if (diff > tolerance) {
                std::cout << "[   ERROR  ] reference = " << ref[i] << ", actual = " << actual[i] << ", diff = " << diff << ", idx = " << i << std::endl;
                return false;
            }
        }
        std::cout << "[   INFO   ] All values of " << string << " are within tolerance." << std::endl;

        return true;
    }

    void SyntheticTestKMeansStudent2_u8c16::run(const KMeansUniversalParams& params) {
        std::cout << "[   INFO   ] " << params << std::endl;
        const auto perf = params.isPerf;
        const auto attempts = params.attempts;
        const auto pointsCount = params.pointsCount;
        const auto clustersCount = std::min(pointsCount, params.clustersCount);

        const int minValue = 0;
        const int maxValue = 128;
        const int dimsNum = 16;
        cv::setNumThreads(1);

        Mat points(pointsCount, dimsNum, CV_32F);
        Mat pointsU8(pointsCount, dimsNum, CV_8U);


        RNG rng(2026);

        cv::Mat initialCenters(clustersCount, dimsNum, CV_32F);
        rng.fill(initialCenters, cv::RNG::UNIFORM, cv::Scalar(minValue), cv::Scalar(maxValue));


        int pointsPerCluster = pointsCount / clustersCount;
        int remainder = pointsCount % clustersCount;

        int pointIdx = 0;
        for (int clusterIdx = 0; clusterIdx < clustersCount; clusterIdx++) {
            int clusterPoints = pointsPerCluster + (clusterIdx < remainder ? 1 : 0);

            for (int i = 0; i < clusterPoints; i++) {

                for (int d = 0; d < dimsNum; d++) {
                    const float centerVal = initialCenters.at<float>(clusterIdx, d);

                    const float noise = rng.gaussian((maxValue - minValue) * 0.05);
                    const float roundPoint = std::max(std::min(std::round(centerVal + noise), 255.f), 0.f);
                    points.at<float>(pointIdx, d) = roundPoint;
                    pointsU8.at<unsigned char>(pointIdx, d) = static_cast<unsigned char>(roundPoint);
                }
                pointIdx++;
            }
        }

        // Перемешиваем точки
        cv::Mat indices(pointsCount, 1, CV_32S);
        for (int i = 0; i < pointsCount; i++) indices.at<int>(i) = i;
        cv::randShuffle(indices);

        cv::Mat pointsShuffled(pointsCount, dimsNum, CV_32F);
        cv::Mat pointsU8Shuffled(pointsCount, dimsNum, CV_8U);

        for (int i = 0; i < pointsCount; i++) {
            int srcIdx = indices.at<int>(i);
            points.row(srcIdx).copyTo(pointsShuffled.row(i));
            pointsU8.row(srcIdx).copyTo(pointsU8Shuffled.row(i));
        }

        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1);

        Mat labelsRef, centersRef, labels, centers;

        int flags = cv::KMEANS_RANDOM_CENTERS;
        if (params.isKMeansPlusPlus) {
            flags = cv::KMEANS_PP_CENTERS;
        }

        setRNGSeed(2026);
        double compactnessRef = kmeans_base(pointsShuffled, clustersCount, labelsRef,
            criteria, attempts, flags, centersRef);

        setRNGSeed(2026);
        double compactnessStud = kmeans_uchar16d(pointsU8Shuffled, clustersCount, labels,
            criteria, attempts, flags, centers);

        std::cout << "compactnessRef: " << compactnessRef << std::endl;
        std::cout << "compactnessStud: " << compactnessStud << std::endl;

        ASSERT_TRUE(checkDiff(centers.ptr<float>(0), centersRef.ptr<float>(0),
            clustersCount * dimsNum, 1.0f, "centers"));
        ASSERT_TRUE(checkDiff(labels.ptr<int32_t>(0), labelsRef.ptr<int32_t>(0),
            pointsCount, 0.f, "labels"));

        EXPECT_NEAR(compactnessStud, compactnessRef, compactnessRef * 0.05);

        if (perf) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < PERF_ITERATIONS; ++i) {
                kmeans_uchar16d(pointsU8Shuffled, clustersCount, labels,
                    criteria, attempts, flags, centers);
            }

            auto end = std::chrono::high_resolution_clock::now();
            const auto duration_st = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Student2 kmeans (16D) took " << duration_st.count() / PERF_ITERATIONS << " ms " << std::endl;

            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < PERF_ITERATIONS; ++i) {
                kmeans_base(pointsShuffled, clustersCount, labels,
                    criteria, attempts, flags, centers);
            }

            end = std::chrono::high_resolution_clock::now();
            const auto duration_cv = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "OpenCV kmeans (16D) took " << duration_cv.count() / PERF_ITERATIONS << " ms " << std::endl;

            const auto st2_ms = duration_st.count() / PERF_ITERATIONS;
            const auto cv_ms = duration_cv.count() / PERF_ITERATIONS;
            const double speedup = static_cast<double>(cv_ms) / static_cast<double>(st2_ms);

            std::cout << "Speedup (CV vs Student2) = " << speedup << std::endl;
        }
    }

    TEST_P(SyntheticTestKMeansStudent2_u8c16, Basic) {
        const KMeansUniversalParams params = GetParam();
        run(params);
    }


    static std::vector<KMeansUniversalParams> synthetic_data_kmeans_student2_u8c16 = {
        { ClustersCount{ 5 }, PointsCount{ 1000 }, Attempts{ 10 },
          Perf{ false }, IsKMeansPlusPlus{ true } },
        { ClustersCount{ 10 }, PointsCount{ 5000 }, Attempts{ 10 },
          Perf{ false }, IsKMeansPlusPlus{ true } },
        { ClustersCount{ 3 }, PointsCount{ 500 }, Attempts{ 10 },
          Perf{ false }, IsKMeansPlusPlus{ true } },
    };


    static std::vector<KMeansUniversalParams> synthetic_data_kmeans_student2_u8c16_perf = {
        { ClustersCount{ 5 }, PointsCount{ 20000 }, Attempts{ 10 },
          Perf{ true }, IsKMeansPlusPlus{ true } },
        { ClustersCount{ 10 }, PointsCount{ 50000 }, Attempts{ 10 },
          Perf{ true }, IsKMeansPlusPlus{ true } },
        { ClustersCount{ 15 }, PointsCount{ 100000 }, Attempts{ 10 },
          Perf{ true }, IsKMeansPlusPlus{ true } },
        { ClustersCount{ 7 }, PointsCount{ 1000000 }, Attempts{ 10 },
          Perf{ true }, IsKMeansPlusPlus{ true } }
    };

    INSTANTIATE_TEST_SUITE_P(Accuracy, SyntheticTestKMeansStudent2_u8c16,
        testing::ValuesIn(synthetic_data_kmeans_student2_u8c16));
    INSTANTIATE_TEST_SUITE_P(Performance, SyntheticTestKMeansStudent2_u8c16,
        testing::ValuesIn(synthetic_data_kmeans_student2_u8c16_perf));
} // namespace