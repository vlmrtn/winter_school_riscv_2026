#include "all_kmeans.h"
#include <opencv2/core/hal/hal.hpp>

namespace stud2 {
    using namespace cv;

    static void generateRandomCenter(int dims, const Vec2f* box, float* center, RNG& rng)
    {
        float margin = 1.f / dims;
        for (int j = 0; j < dims; j++)
            center[j] = ((float)rng * (1.f + margin * 2.f) - margin) * (box[j][1] - box[j][0]) + box[j][0];
    }

    class KMeansPPDistanceComputer : public ParallelLoopBody
    {
    public:
        KMeansPPDistanceComputer(float* tdist2_, const Mat& data_, const float* dist_, int ci_, int dims_) :
            tdist2(tdist2_), data(data_), dist(dist_), ci(ci_), dims(dims_)
        {
        }

        void operator()(const cv::Range& range) const CV_OVERRIDE
        {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++)
            {
                const unsigned char* dI = data.ptr<unsigned char>(i);
                const unsigned char* dCI = data.ptr<unsigned char>(ci);
                float L2NormSqr = 0.0f;
                for (int j = 0; j < dims; j+=4) {
                    float diff = static_cast<float>(dI[j]) - static_cast<float>(dCI[j]);
                    L2NormSqr += diff * diff;

                    float diff = static_cast<float>(dI[j+1]) - static_cast<float>(dCI[j+1]);
                    L2NormSqr += diff * diff;

                    float diff = static_cast<float>(dI[j+2]) - static_cast<float>(dCI[j+2]);
                    L2NormSqr += diff * diff;

                    float diff = static_cast<float>(dI[j+3]) - static_cast<float>(dCI[j+3]);
                    L2NormSqr += diff * diff;
                }
                tdist2[i] = std::min(dist[i], L2NormSqr);
            }
        }

    private:
        KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&);

        float* tdist2;
        const Mat& data;
        const float* dist;
        const int ci;
        const int dims;
    };

    static void generateCentersPP(const Mat& data, Mat& _out_centers,
        int K, RNG& rng, int trials)
    {
        const int dims = data.cols, N = data.rows;
        cv::AutoBuffer<int, 64> _centers(K);
        int* centers = &_centers[0];
        cv::AutoBuffer<float, 0> _dist(N * 3); // dist, tdist, tdist2
        float* dist = &_dist[0], * tdist = dist + N, * tdist2 = tdist + N;
        double sum0 = 0;

        centers[0] = (unsigned)rng % N;
        const unsigned char* dCI = data.ptr<unsigned char>(centers[0]);

        for (int i = 0; i < N; i++)
        {
            const unsigned char* dI = data.ptr<unsigned char>(i);
            float dist_val = 0.0f;
            for (int j = 0; j < dims; j+=4) {
                float diff = static_cast<float>(dI[j]) - static_cast<float>(dCI[j]);
                dist_val += diff * diff;

                float diff = static_cast<float>(dI[j+1]) - static_cast<float>(dCI[j+1]);
                dist_val += diff * diff;

                float diff = static_cast<float>(dI[j+2]) - static_cast<float>(dCI[j+2]);
                dist_val += diff * diff;

                float diff = static_cast<float>(dI[j+3]) - static_cast<float>(dCI[j+3]);
                dist_val += diff * diff;
            }
            dist[i] = dist_val;
            sum0 += dist[i];c
        }

        for (int k = 1; k < K; k++)
        {
            double bestSum = DBL_MAX;
            int bestCenter = -1;

            for (int j = 0; j < trials; j++)
            {
                double p = (double)rng * sum0;
                int ci = 0;
                for (; ci < N - 1; ci++)
                {
                    p -= dist[ci];
                    if (p <= 0)
                        break;
                }

                parallel_for_(Range(0, N),
                    KMeansPPDistanceComputer(tdist2, data, dist, ci, dims));
                double s = 0;
                int i = 0;
#if CV_ENABLE_UNROLLED
                for (; i + 7 < N; i += 8)
                {
                    s += tdist2[i + 0] + tdist2[i + 1] + tdist2[i + 2] + tdist2[i + 3] +
                        tdist2[i + 4] + tdist2[i + 5] + tdist2[i + 6] + tdist2[i + 7];
                }
#endif
                size_t vl1 = __riscv_vsetvl_e32m4(N);
                vfloat32m4_t temp = __riscv_vfmv_v_f_f32m4(0.f, vl1);
                vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.f, vl1);

                for (; i < N; i+=vl1)
                {
                    size_t vl = __riscv_vsetvl_e32m4(N-i);
                    
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(tdist2, vl);
                    temp = __riscv_vadd_vv_f32m4(temp, v1, vl);
       
                }
                acc = __riscv_vfredosum_vs_f32m4_f32m1(temp, acc, vl1);
                s += __riscv_vfmv_f_s_f32m1_f32(acc);

                if (s < bestSum)
                {
                    bestSum = s;
                    bestCenter = ci;
                    std::swap(tdist, tdist2);
                }
            }
            if (bestCenter < 0)
                CV_Error(Error::StsNoConv, "kmeans: can't update cluster center (check input for huge or NaN values)");
            centers[k] = bestCenter;
            sum0 = bestSum;
            std::swap(dist, tdist);
        }

        for (int k = 0; k < K; k++)
        {
            const unsigned char* src = data.ptr<unsigned char>(centers[k]);
            float* dst = _out_centers.ptr<float>(k);
            for (int j = 0; j < dims; j += 4)
            {
                dst[j] = static_cast<float>(src[j]);

                dst[j+1] = static_cast<float>(src[j+1]);

                dst[j+2] = static_cast<float>(src[j+2]);

                dst[j+3] = static_cast<float>(src[j+3]);
            }
        }
    }

    template<bool onlyDistance>
    class KMeansDistanceComputer : public ParallelLoopBody
    {
    public:
        KMeansDistanceComputer(double* distances_,
            unsigned char* labels_,
            const Mat& data_,
            const Mat& centers_,
            const int K_,
            const int dims_)
            : distances(distances_),
            labels(labels_),
            data(data_),
            centers(centers_),
            K(K_),
            dims(dims_)
        {
        }

        void operator()(const Range& range) const CV_OVERRIDE
        {
            const int begin = range.start;
            const int end = range.end;



            for (int i = begin; i < end; ++i)
            {
                const unsigned char* sample = data.ptr<unsigned char>(i);
                if (onlyDistance)
                {
                    const float* center = centers.ptr<float>(labels[i]);
                    double dist_val = 0.0;
                    for (int j = 0; j < dims; j+=4) {
                        double diff = static_cast<double>(sample[j]) - static_cast<double>(center[j]);
                        dist_val += diff * diff;

                        double diff = static_cast<double>(sample[j+1]) - static_cast<double>(center[j+1]);
                        dist_val += diff * diff;

                        double diff = static_cast<double>(sample[j+2]) - static_cast<double>(center[j+2]);
                        dist_val += diff * diff;

                        double diff = static_cast<double>(sample[j+3]) - static_cast<double>(center[j+3]);
                        dist_val += diff * diff;
                    }
                    distances[i] = dist_val;
                    continue;
                }
                else
                {
                    unsigned char k_best = 0;
                    double min_dist = DBL_MAX;

                    for (int k = 0; k < K; k++)
                    {
                        const float* center = centers.ptr<float>(k);
                        double dist_val = 0.0;
                        for (int j = 0; j < dims; j+=4) {
                            double diff = static_cast<double>(sample[j]) - static_cast<double>(center[j]);
                            dist_val += diff * diff;

                            double diff = static_cast<double>(sample[j+1]) - static_cast<double>(center[j+1]);
                            dist_val += diff * diff;

                            double diff = static_cast<double>(sample[j+2]) - static_cast<double>(center[j+2]);
                            dist_val += diff * diff;

                            double diff = static_cast<double>(sample[j+3]) - static_cast<double>(center[j+3]);
                            dist_val += diff * diff;
                        }

                        if (min_dist > dist_val)
                        {
                            min_dist = dist_val;
                            k_best = static_cast<unsigned char>(k);
                        }
                    }

                    distances[i] = min_dist;
                    labels[i] = k_best;
                }
            }
        }

    private:
        KMeansDistanceComputer& operator=(const KMeansDistanceComputer&);

        double* distances;
        unsigned char* labels;
        const Mat& data;
        const Mat& centers;
        const int K;
        const int dims;
    };


    double kmeans(InputArray _data, int K,
        InputOutputArray _bestLabels,
        TermCriteria criteria, int attempts,
        int flags, OutputArray _centers)
    {
        const int SPP_TRIALS = 3;
        Mat data0 = _data.getMat();
        const bool isrow = data0.rows == 1;
        const int N = isrow ? data0.cols : data0.rows;
        const int dims = (isrow ? 1 : data0.cols) * data0.channels();
        const int centersType = CV_32F;

        attempts = std::max(attempts, 1);
        CV_Assert(data0.dims <= 2 && data0.depth() == CV_8U && K > 0 && K <= 255);
        CV_CheckGE(N, K, "There can't be more clusters than elements");

        Mat data(N, dims, CV_8U, data0.ptr(), isrow ? dims * sizeof(unsigned char) : static_cast<size_t>(data0.step));

        _bestLabels.create(N, 1, CV_32S, -1, true);

        Mat _labels, best_labels = _bestLabels.getMat();

        if (flags & KMEANS_USE_INITIAL_LABELS)
        {
            CV_Assert((best_labels.cols == 1 || best_labels.rows == 1) &&
                best_labels.cols * best_labels.rows == N &&
                best_labels.type() == CV_32S &&
                best_labels.isContinuous());

            // Convert int labels to unsigned char for internal use
            _labels.create(N, 1, CV_8U);
            Mat int_labels = best_labels.reshape(1, N);
            for (int i = 0; i < N; i++) {
                int label_val = int_labels.at<int>(i);
                CV_Assert(label_val >= 0 && label_val < K);
                _labels.at<unsigned char>(i) = static_cast<unsigned char>(label_val);
            }
        }
        else
        {
            if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
                best_labels.cols * best_labels.rows == N &&
                best_labels.isContinuous()))
            {
                _bestLabels.create(N, 1, CV_32S);
                best_labels = _bestLabels.getMat();
            }
            _labels.create(N, 1, CV_8U);
        }
        unsigned char* labels = _labels.ptr<unsigned char>();

        Mat centers(K, dims, centersType), old_centers(K, dims, centersType), temp(1, dims, centersType);
        cv::AutoBuffer<int, 64> counters(K);
        cv::AutoBuffer<double, 64> dists(N);
        RNG& rng = theRNG();

        if (criteria.type & TermCriteria::EPS)
            criteria.epsilon = std::max(criteria.epsilon, 0.);
        else
            criteria.epsilon = FLT_EPSILON;
        criteria.epsilon *= criteria.epsilon;

        if (criteria.type & TermCriteria::COUNT)
            criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
        else
            criteria.maxCount = 100;

        if (K == 1)
        {
            attempts = 1;
            criteria.maxCount = 2;
        }

        cv::AutoBuffer<Vec2f, 64> box(dims);
        if (!(flags & KMEANS_PP_CENTERS))
        {
            {
                const unsigned char* sample = data.ptr<unsigned char>(0);
                for (int j = 0; j < dims; j+=4) {
                    float val = static_cast<float>(sample[j]);
                    box[j] = Vec2f(val, val);

                    val = static_cast<float>(sample[j+1]);
                    box[j+1] = Vec2f(val, val);

                    val = static_cast<float>(sample[j+2]);
                    box[j+2] = Vec2f(val, val);

                    val = static_cast<float>(sample[j+3]);
                    box[j+3] = Vec2f(val, val);
                }
            }
            for (int i = 1; i < N; i++)
            {
                const unsigned char* sample = data.ptr<unsigned char>(i);
                for (int j = 0; j < dims; j+=4)
                {
                    float v = static_cast<float>(sample[j]);
                    box[j][0] = std::min(box[j][0], v);
                    box[j][1] = std::max(box[j][1], v);

                    v = static_cast<float>(sample[j+1]);
                    box[j+1][0] = std::min(box[j+1][0], v);
                    box[j+1][1] = std::max(box[j+1][1], v);

                    v = static_cast<float>(sample[j+2]);
                    box[j+2][0] = std::min(box[j+2][0], v);
                    box[j+2][1] = std::max(box[j+2][1], v);

                    v = static_cast<float>(sample[j+3]);
                    box[j+3][0] = std::min(box[j+3][0], v);
                    box[j+3][1] = std::max(box[j+3][1], v);
                }
            }
        }

        double best_compactness = DBL_MAX;
        for (int a = 0; a < attempts; a++)
        {
            double compactness = 0;

            for (int iter = 0; ;)
            {
                double max_center_shift = iter == 0 ? DBL_MAX : 0.0;

                swap(centers, old_centers);

                if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)))
                {
                    if (flags & KMEANS_PP_CENTERS)
                        generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                    else
                    {
                        for (int k = 0; k < K; k++)
                            generateRandomCenter(dims, box.data(), centers.ptr<float>(k), rng);
                    }
                }
                else
                {
                    // compute centers
                    centers = Scalar(0);
                    for (int k = 0; k < K; k++)
                        counters[k] = 0;

                    for (int i = 0; i < N; i++)
                    {
                        const unsigned char* sample = data.ptr<unsigned char>(i);
                        int k = labels[i];
                        float* center = centers.ptr<float>(k);
                        for (int j = 0; j < dims; j+=4)
                        {
                            center[j] += static_cast<float>(sample[j]);

                            center[j+1] += static_cast<float>(sample[j+1]);

                            center[j+2] += static_cast<float>(sample[j+2]);

                            center[j+3] += static_cast<float>(sample[j+3]);
                        }
                        counters[k]++;
                    }

                    for (int k = 0; k < K; k++)
                    {
                        if (counters[k] != 0)
                            continue;

                        // if some cluster appeared to be empty then:
                        //   1. find the biggest cluster
                        //   2. find the farthest from the center point in the biggest cluster
                        //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                        int max_k = 0;
                        for (int k1 = 1; k1 < K; k1++)
                        {
                            if (counters[max_k] < counters[k1])
                                max_k = k1;
                        }

                        double max_dist = 0;
                        int farthest_i = -1;
                        float* base_center = centers.ptr<float>(max_k);
                        float* _base_center = temp.ptr<float>(); // normalized
                        float scale = 1.f / counters[max_k];
                        for (int j = 0; j < dims; j += 4)
                        {
                            _base_center[j] = base_center[j] * scale;

                            _base_center[j+1] = base_center[j+1] * scale;

                            _base_center[j+2] = base_center[j+2] * scale;

                            _base_center[j+3] = base_center[j+3] * scale;
                        }
                        for (int i = 0; i < N; i++)
                        {
                            if (labels[i] != max_k)
                                continue;
                            const unsigned char* sample = data.ptr<unsigned char>(i);
                            double dist_val = 0.0;
                            for (int j = 0; j < dims; j+=4) {
                                double diff = static_cast<double>(sample[j]) - static_cast<double>(_base_center[j]);
                                dist_val += diff * diff;

                                diff = static_cast<double>(sample[j+1]) - static_cast<double>(_base_center[j+1]);
                                dist_val += diff * diff;

                                diff = static_cast<double>(sample[j+2]) - static_cast<double>(_base_center[j+2]);
                                dist_val += diff * diff;

                                diff = static_cast<double>(sample[j+3]) - static_cast<double>(_base_center[j+3]);
                                dist_val += diff * diff;
                            }

                            if (max_dist <= dist_val)
                            {
                                max_dist = dist_val;
                                farthest_i = i;
                            }
                        }

                        counters[max_k]--;
                        counters[k]++;
                        labels[farthest_i] = static_cast<unsigned char>(k);

                        const unsigned char* sample = data.ptr<unsigned char>(farthest_i);
                        float* cur_center = centers.ptr<float>(k);
                        for (int j = 0; j < dims; j+=4)
                        {
                            base_center[j] -= static_cast<float>(sample[j]);
                            cur_center[j] += static_cast<float>(sample[j]);

                            base_center[j+1] -= static_cast<float>(sample[j+1]);
                            cur_center[j+1] += static_cast<float>(sample[j+1]);

                            base_center[j+2] -= static_cast<float>(sample[j+2]);
                            cur_center[j+2] += static_cast<float>(sample[j+2]);

                            base_center[j+3] -= static_cast<float>(sample[j+3]);
                            cur_center[j+3] += static_cast<float>(sample[j+3]);
                        }
                    }

                    for (int k = 0; k < K; k++)
                    {
                        float* center = centers.ptr<float>(k);
                        CV_Assert(counters[k] != 0);

                        float scale = 1.f / counters[k];
                        for (int j = 0; j < dims; j+=4)
                        {
                            center[j] *= scale;

                            center[j+1] *= scale;

                            center[j+2] *= scale;

                            center[j+3] *= scale;
                        }

                        if (iter > 0)
                        {
                            double dist = 0;
                            const float* old_center = old_centers.ptr<float>(k);
                            for (int j = 0; j < dims; j+=4)
                            {
                                double t = center[j] - old_center[j];
                                dist += t * t;

                                double t = center[j+1] - old_center[j+1];
                                dist += t * t;

                                double t = center[j+2] - old_center[j+2];
                                dist += t * t;

                                double t = center[j+3] - old_center[j+3];
                                dist += t * t;

                     
                            }
                            max_center_shift = std::max(max_center_shift, dist);
                        }
                    }
                }

                bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);

                if (isLastIter)
                {
                    parallel_for_(Range(0, N), KMeansDistanceComputer<true>(dists.data(), labels, data, centers, K, dims));
                    compactness = sum(Mat(Size(N, 1), CV_64F, &dists[0]))[0];
                    break;
                }
                else
                {
                    parallel_for_(Range(0, N), KMeansDistanceComputer<false>(dists.data(), labels, data, centers, K, dims));
                }
            }

            if (compactness < best_compactness)
            {
                best_compactness = compactness;
                if (_centers.needed())
                {
                    if (_centers.fixedType() && _centers.channels() == dims)
                        centers.reshape(dims).copyTo(_centers);
                    else
                        centers.copyTo(_centers);
                }

                // Convert internal uchar labels back to int
                Mat int_labels = best_labels.reshape(1, N);
                for (int i = 0; i < N; i++) {
                    int_labels.at<int>(i) = static_cast<int>(labels[i]);
                }
            }
        }

        return best_compactness;
    }
} // namespace stud2

double kmeans_uchar(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
) {
    return stud2::kmeans(data, K, bestLabels, criteria, attempts, flags, centers);
}