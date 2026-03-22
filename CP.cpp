#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct Point {
    std::size_t id;
    double x;
    double y;
};

struct ClosestPairResult {
    double bestDistSq = std::numeric_limits<double>::infinity();
    Point a {};
    Point b {};
    std::uint64_t comparisons = 0;
};

/**
 * @brief Computes squared Euclidean distance between two points.
 *
 * @param p1 First point.
 * @param p2 Second point.
 * @return double Squared distance between p1 and p2.
 */
double distance_squared_between(const Point& p1, const Point& p2)
{
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

/**
 * @brief Atomically lowers the shared best-distance bound when a better value is found.
 *
 * @param globalBestDistSq Shared best squared distance across workers.
 * @param candidateDistSq Candidate squared distance to publish.
 */
void publish_global_min(std::atomic<double>& globalBestDistSq, double candidateDistSq)
{
    double observed = globalBestDistSq.load(std::memory_order_relaxed);
    while (candidateDistSq < observed && !globalBestDistSq.compare_exchange_weak(observed, candidateDistSq, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

/**
 * @brief Evaluates one point pair and updates the running closest-pair result.
 *
 * @param p First point.
 * @param q Second point.
 * @param result In/out aggregate result for best distance, best pair, and comparisons.
 */
void consider_pair(const Point& p, const Point& q, ClosestPairResult& result)
{
    ++result.comparisons;
    const double dSq = distance_squared_between(p, q);
    if (dSq < result.bestDistSq) {
        result.bestDistSq = dSq;
        result.a = p;
        result.b = q;
    }
}

/**
 * @brief Generates synthetic 2D points with controllable clumpiness.
 *
 * @param count Number of points to generate.
 * @param seed RNG seed for reproducible datasets.
 * @param distributionModifier Distribution control in [0, 10], where 10 is even and 0 is clumpy.
 * @return std::vector<Point> Generated point set.
 */
std::vector<Point> generate_points(
    std::size_t count,
    unsigned int seed,
    double distributionModifier)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniformDist(0.0, 100000.0);

    std::vector<Point> points;
    points.reserve(count);

    // 10.0 means near-uniform/even distribution.
    if (distributionModifier >= 10.0 - 1e-12) {
        for (std::size_t i = 0; i < count; ++i) {
            points.push_back(Point { i, uniformDist(rng), uniformDist(rng) });
        }
        return points;
    }

    // 0.0 means super-clumpy: very few cluster centers and tight spread.
    const double normalized = std::clamp(distributionModifier / 10.0, 0.0, 1.0);
    const std::size_t clusterCount = 1 + static_cast<std::size_t>(std::llround(normalized * 49.0)); // [1, 50]

    std::vector<std::pair<double, double>> centers;
    centers.reserve(clusterCount);
    for (std::size_t c = 0; c < clusterCount; ++c) {
        centers.push_back(std::make_pair(uniformDist(rng), uniformDist(rng)));
    }

    const double minSigma = 150.0;
    const double maxSigma = 12000.0;
    const double sigma = minSigma + normalized * (maxSigma - minSigma);

    std::normal_distribution<double> clusterOffset(0.0, sigma);
    std::uniform_int_distribution<std::size_t> pickCluster(0, clusterCount - 1);
    std::bernoulli_distribution injectUniformNoise(normalized * 0.15);

    for (std::size_t i = 0; i < count; ++i) {
        if (injectUniformNoise(rng)) {
            points.push_back(Point { i, uniformDist(rng), uniformDist(rng) });
            continue;
        }

        const auto& center = centers[pickCluster(rng)];
        const double x = std::clamp(center.first + clusterOffset(rng), 0.0, 100000.0);
        const double y = std::clamp(center.second + clusterOffset(rng), 0.0, 100000.0);

        points.push_back(Point { i, x, y });
    }

    return points;
}

/**
 * @brief Solves closest-pair using full O(n^2) brute force.
 *
 * @param points Input points.
 * @return ClosestPairResult Closest-pair result and comparison count.
 */
ClosestPairResult brute_force_closest_pair(const std::vector<Point>& points)
{
    ClosestPairResult result;

    if (points.size() < 2) {
        return result;
    }

    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            consider_pair(points[i], points[j], result);
        }
    }

    return result;
}

/**
 * @brief Builds an initial distance bound from adjacent pairs in X- and Y-sorted orders.
 *
 * @param px Points sorted by X (then Y).
 * @param py Points sorted by Y (then X).
 * @return ClosestPairResult Initial bound and seeded best pair.
 */
ClosestPairResult bootstrap_initial_bound(
    const std::vector<Point>& px,
    const std::vector<Point>& py)
{
    ClosestPairResult result;

    if (px.size() < 2) {
        return result;
    }

    for (std::size_t i = 0; i + 1 < px.size(); ++i) {
        consider_pair(px[i], px[i + 1], result);
    }

    for (std::size_t i = 0; i + 1 < py.size(); ++i) {
        consider_pair(py[i], py[i + 1], result);
    }

    return result;
}

/**
 * @brief Merges left/right D&C subproblem results and checks the cross-strip region.
 *
 * @param leftResult Closest-pair result from the left half.
 * @param rightResult Closest-pair result from the right half.
 * @param py Points for this subproblem sorted by Y.
 * @param splitX X coordinate of the split line.
 * @return ClosestPairResult Merged closest-pair result for this subproblem.
 */
ClosestPairResult merge_divide_and_conquer_results(
    const ClosestPairResult& leftResult,
    const ClosestPairResult& rightResult,
    const std::vector<Point>& py,
    double splitX)
{
    ClosestPairResult merged = (leftResult.bestDistSq <= rightResult.bestDistSq) ? leftResult : rightResult;
    merged.comparisons = leftResult.comparisons + rightResult.comparisons;

    std::vector<Point> strip;
    strip.reserve(py.size());

    for (const Point& p : py) {
        const double dx = p.x - splitX;
        if ((dx * dx) < merged.bestDistSq) {
            strip.push_back(p);
        }
    }

    for (std::size_t i = 0; i < strip.size(); ++i) {
        const std::size_t upper = std::min<std::size_t>(strip.size(), i + 8);
        for (std::size_t j = i + 1; j < upper; ++j) {
            const double dy = strip[j].y - strip[i].y;
            if ((dy * dy) >= merged.bestDistSq) {
                break;
            }

            consider_pair(strip[i], strip[j], merged);
        }
    }

    return merged;
}

/**
 * @brief Runs brute-force closest-pair over a half-open index range in X-sorted data.
 *
 * @param px Points sorted by X.
 * @param beginIndex Inclusive begin index.
 * @param endIndex Exclusive end index.
 * @return ClosestPairResult Closest-pair result for the subrange.
 */
ClosestPairResult brute_force_closest_pair_range(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex)
{
    ClosestPairResult result;

    if (endIndex <= beginIndex + 1) {
        return result;
    }

    for (std::size_t i = beginIndex; i < endIndex; ++i) {
        for (std::size_t j = i + 1; j < endIndex; ++j) {
            consider_pair(px[i], px[j], result);
        }
    }

    return result;
}

/**
 * @brief Splits a Y-sorted subproblem into left/right Y-sorted vectors using X-rank membership.
 *
 * @param py Points sorted by Y for current subproblem.
 * @param idToRank Mapping from point ID to rank in global X-sorted order.
 * @param splitRank Rank threshold that separates left and right halves.
 * @param leftPy Output Y-sorted points belonging to left half.
 * @param rightPy Output Y-sorted points belonging to right half.
 */
void split_py_with_rank_cut(
    const std::vector<Point>& py,
    const std::vector<std::size_t>& idToRank,
    std::size_t splitRank,
    std::vector<Point>& leftPy,
    std::vector<Point>& rightPy)
{
    leftPy.clear();
    rightPy.clear();
    leftPy.reserve(py.size() / 2 + 1);
    rightPy.reserve(py.size() / 2 + 1);

    for (const Point& p : py) {
        if (idToRank[p.id] < splitRank) {
            leftPy.push_back(p);
        } else {
            rightPy.push_back(p);
        }
    }
}

/**
 * @brief Recursive serial divide-and-conquer closest-pair solver.
 *
 * @param px Points sorted by X.
 * @param beginIndex Inclusive begin index in px for this subproblem.
 * @param endIndex Exclusive end index in px for this subproblem.
 * @param py Points in this subproblem sorted by Y.
 * @param idToRank Mapping from point ID to rank in px.
 * @return ClosestPairResult Closest-pair result for this subproblem.
 */
ClosestPairResult divide_and_conquer_serial_rec(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex,
    const std::vector<Point>& py,
    const std::vector<std::size_t>& idToRank)
{
    if (endIndex - beginIndex <= 3) {
        return brute_force_closest_pair_range(px, beginIndex, endIndex);
    }

    const std::size_t mid = beginIndex + (endIndex - beginIndex) / 2;
    const double splitX = px[mid].x;

    std::vector<Point> leftPy;
    std::vector<Point> rightPy;
    split_py_with_rank_cut(py, idToRank, mid, leftPy, rightPy);

    const ClosestPairResult leftResult = divide_and_conquer_serial_rec(px, beginIndex, mid, leftPy, idToRank);
    const ClosestPairResult rightResult = divide_and_conquer_serial_rec(px, mid, endIndex, rightPy, idToRank);

    return merge_divide_and_conquer_results(leftResult, rightResult, py, splitX);
}

/**
 * @brief Recursive parallel divide-and-conquer closest-pair solver.
 *
 * @param px Points sorted by X.
 * @param beginIndex Inclusive begin index in px for this subproblem.
 * @param endIndex Exclusive end index in px for this subproblem.
 * @param py Points in this subproblem sorted by Y.
 * @param idToRank Mapping from point ID to rank in px.
 * @param threadBudget Maximum threads this call and descendants may consume.
 * @param parallelThreshold Minimum subproblem size to spawn parallel subcalls.
 * @return ClosestPairResult Closest-pair result for this subproblem.
 */
ClosestPairResult divide_and_conquer_parallel_rec(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex,
    const std::vector<Point>& py,
    const std::vector<std::size_t>& idToRank,
    std::size_t threadBudget,
    std::size_t parallelThreshold)
{
    if (endIndex - beginIndex <= 3) {
        return brute_force_closest_pair_range(px, beginIndex, endIndex);
    }

    const std::size_t mid = beginIndex + (endIndex - beginIndex) / 2;
    const double splitX = px[mid].x;

    std::vector<Point> leftPy;
    std::vector<Point> rightPy;
    split_py_with_rank_cut(py, idToRank, mid, leftPy, rightPy);

    ClosestPairResult leftResult;
    ClosestPairResult rightResult;

    const bool canParallelize = (threadBudget > 1) && ((endIndex - beginIndex) >= parallelThreshold);
    if (canParallelize) {
        const std::size_t leftBudget = threadBudget / 2;
        const std::size_t rightBudget = threadBudget - leftBudget;

        auto leftFuture = std::async(
            std::launch::async,
            [&px, &idToRank, beginIndex, mid, leftBudget, parallelThreshold, leftPy = std::move(leftPy)]() {
                return divide_and_conquer_parallel_rec(
                    px,
                    beginIndex,
                    mid,
                    leftPy,
                    idToRank,
                    leftBudget,
                    parallelThreshold);
            });

        rightResult = divide_and_conquer_parallel_rec(
            px,
            mid,
            endIndex,
            rightPy,
            idToRank,
            rightBudget,
            parallelThreshold);
        leftResult = leftFuture.get();
    } else {
        leftResult = divide_and_conquer_parallel_rec(
            px,
            beginIndex,
            mid,
            leftPy,
            idToRank,
            1,
            parallelThreshold);
        rightResult = divide_and_conquer_parallel_rec(
            px,
            mid,
            endIndex,
            rightPy,
            idToRank,
            1,
            parallelThreshold);
    }

    return merge_divide_and_conquer_results(leftResult, rightResult, py, splitX);
}

/**
 * @brief Top-level serial divide-and-conquer closest-pair solver.
 *
 * @param points Input points.
 * @return ClosestPairResult Closest-pair result computed by serial D&C.
 */
ClosestPairResult divide_and_conquer_serial_closest_pair(const std::vector<Point>& points)
{
    ClosestPairResult result;

    if (points.size() < 2) {
        return result;
    }

    std::vector<Point> px = points;
    std::vector<Point> py = points;

    std::sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.id < b.id;
    });

    std::sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.id < b.id;
    });

    std::vector<std::size_t> idToRank(points.size(), 0);
    for (std::size_t i = 0; i < px.size(); ++i) {
        idToRank[px[i].id] = i;
    }

    return divide_and_conquer_serial_rec(px, 0, px.size(), py, idToRank);
}

/**
 * @brief Top-level parallel divide-and-conquer closest-pair solver.
 *
 * @param points Input points.
 * @param threadCount Number of threads to use.
 * @return ClosestPairResult Closest-pair result computed by parallel D&C.
 */
ClosestPairResult divide_and_conquer_parallel_closest_pair(
    const std::vector<Point>& points,
    std::size_t threadCount)
{
    ClosestPairResult result;

    if (points.size() < 2) {
        return result;
    }

    if (threadCount == 0) {
        threadCount = 1;
    }

    std::vector<Point> px = points;
    std::vector<Point> py = points;

    std::sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.id < b.id;
    });

    std::sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.id < b.id;
    });

    std::vector<std::size_t> idToRank(points.size(), 0);
    for (std::size_t i = 0; i < px.size(); ++i) {
        idToRank[px[i].id] = i;
    }

    const std::size_t parallelThreshold = 2048;
    return divide_and_conquer_parallel_rec(
        px,
        0,
        px.size(),
        py,
        idToRank,
        threadCount,
        parallelThreshold);
}

/**
 * @brief Runs serial CRNS search on pre-sorted subproblem views.
 *
 * @param px Points sorted by X.
 * @param beginIndex Inclusive begin index in px.
 * @param endIndex Exclusive end index in px.
 * @param py Same subproblem points sorted by Y.
 * @return ClosestPairResult Closest-pair result for the subproblem.
 */
ClosestPairResult crns_serial_search_from_sorted_views(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex,
    const std::vector<Point>& py)
{
    ClosestPairResult result;

    if (endIndex <= beginIndex + 1) {
        return result;
    }

    for (std::size_t i = beginIndex; i + 1 < endIndex; ++i) {
        consider_pair(px[i], px[i + 1], result);
    }

    for (std::size_t i = 0; i + 1 < py.size(); ++i) {
        consider_pair(py[i], py[i + 1], result);
    }

    for (std::size_t i = beginIndex; i < endIndex; ++i) {
        const Point& p = px[i];

        std::size_t j = i + 1;
        while (j < endIndex) {
            const double dx = px[j].x - p.x;
            if ((dx * dx) >= result.bestDistSq) {
                break;
            }

            const Point& q = px[j];
            const double dy = q.y - p.y;
            if ((dy * dy) < result.bestDistSq) {
                consider_pair(p, q, result);
            }

            ++j;
        }
    }

    return result;
}

/**
 * @brief Hybrid recursive closest-pair solver using D&C outside and CRNS at leaf regions.
 *
 * @param px X Sorted points
 * @param beginIndex Inclusive begin index in px for this subproblem
 * @param endIndex Exclusive end index in px for this subproblem
 * @param py Y Sorted points
 * @param idToRank Mapping from point ID to its rank in the X-sorted order (index in px)
 * @param threadBudget Number of threads that this call can use (including subcalls). If 1, runs serially.
 * @param parallelThreshold Minimum problem size (number of points in the current subproblem) to consider parallelizing. Below this, runs serially even if threadBudget > 1.
 * @param crnsLeafThreshold Minimum problem size to switch to CRNS leaf search. If the number of points in the current subproblem is <= this, runs CRNS instead of further dividing, even if threadBudget and parallelThreshold would allow further division.
 * @return ClosestPairResult
 */
ClosestPairResult hybrid_parallel_dnc_crns_rec(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex,
    const std::vector<Point>& py,
    const std::vector<std::size_t>& idToRank,
    std::size_t threadBudget,
    std::size_t parallelThreshold,
    std::size_t crnsLeafThreshold)
{
    const std::size_t problemSize = endIndex - beginIndex;

    if (problemSize <= 3) {
        return brute_force_closest_pair_range(px, beginIndex, endIndex);
    }

    if (problemSize <= crnsLeafThreshold) {
        return crns_serial_search_from_sorted_views(px, beginIndex, endIndex, py);
    }

    const std::size_t mid = beginIndex + problemSize / 2;
    const double splitX = px[mid].x;

    std::vector<Point> leftPy;
    std::vector<Point> rightPy;
    split_py_with_rank_cut(py, idToRank, mid, leftPy, rightPy);

    ClosestPairResult leftResult;
    ClosestPairResult rightResult;

    const bool canParallelize = (threadBudget > 1) && (problemSize >= parallelThreshold);
    if (canParallelize) {
        const std::size_t leftBudget = threadBudget / 2;
        const std::size_t rightBudget = threadBudget - leftBudget;

        auto leftFuture = std::async(
            std::launch::async,
            [&px, &idToRank, beginIndex, mid, leftBudget, parallelThreshold, crnsLeafThreshold, leftPy = std::move(leftPy)]() {
                return hybrid_parallel_dnc_crns_rec(
                    px,
                    beginIndex,
                    mid,
                    leftPy,
                    idToRank,
                    leftBudget,
                    parallelThreshold,
                    crnsLeafThreshold);
            });

        rightResult = hybrid_parallel_dnc_crns_rec(
            px,
            mid,
            endIndex,
            rightPy,
            idToRank,
            rightBudget,
            parallelThreshold,
            crnsLeafThreshold);
        leftResult = leftFuture.get();
    } else {
        leftResult = hybrid_parallel_dnc_crns_rec(
            px,
            beginIndex,
            mid,
            leftPy,
            idToRank,
            1,
            parallelThreshold,
            crnsLeafThreshold);
        rightResult = hybrid_parallel_dnc_crns_rec(
            px,
            mid,
            endIndex,
            rightPy,
            idToRank,
            1,
            parallelThreshold,
            crnsLeafThreshold);
    }

    return merge_divide_and_conquer_results(leftResult, rightResult, py, splitX);
}

/**
 * @brief Top-level entry for the hybrid solver (parallel D&C with CRNS leaves).
 *
 * @param points Input points.
 * @param threadCount Number of threads to use.
 * @return ClosestPairResult Closest-pair result computed by the hybrid method.
 */
ClosestPairResult hybrid_parallel_dnc_crns_closest_pair(
    const std::vector<Point>& points,
    std::size_t threadCount)
{
    ClosestPairResult result;

    if (points.size() < 2) {
        return result;
    }

    if (threadCount == 0) {
        threadCount = 1;
    }

    std::vector<Point> px = points;
    std::vector<Point> py = points;

    std::sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.id < b.id;
    });

    std::sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.id < b.id;
    });

    std::vector<std::size_t> idToRank(points.size(), 0);
    for (std::size_t i = 0; i < px.size(); ++i) {
        idToRank[px[i].id] = i;
    }

    const std::size_t parallelThreshold = 4096;
    const std::size_t crnsLeafThreshold = 2048;
    return hybrid_parallel_dnc_crns_rec(
        px,
        0,
        px.size(),
        py,
        idToRank,
        threadCount,
        parallelThreshold,
        crnsLeafThreshold);
}

/**
 * @brief Serial CRNS (Cross-Rank Neighbor Search) closest-pair solver.
 *
 * @param points Input points.
 * @return ClosestPairResult Closest-pair result computed by serial CRNS.
 */
ClosestPairResult cross_rank_serial_search(const std::vector<Point>& points)
{
    ClosestPairResult finalResult;

    if (points.size() < 2) {
        return finalResult;
    }

    std::vector<Point> px = points;
    std::vector<Point> py = points;

    std::sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.id < b.id;
    });

    std::sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.id < b.id;
    });

    finalResult = bootstrap_initial_bound(px, py);

    for (std::size_t i = 0; i < px.size(); ++i) {
        const Point& p = px[i];

        std::size_t j = i + 1;
        while (j < px.size()) {
            const double dx = px[j].x - p.x;
            if ((dx * dx) >= finalResult.bestDistSq) {
                break;
            }

            const Point& q = px[j];
            const double dy = q.y - p.y;
            if ((dy * dy) < finalResult.bestDistSq) {
                consider_pair(p, q, finalResult);
            }

            ++j;
        }
    }

    return finalResult;
}

/**
 * @brief Worker for one chunk of the CRNS X-forward outer loop.
 *
 * @param px Points sorted by X.
 * @param beginIndex Inclusive begin index for this worker's chunk.
 * @param endIndex Exclusive end index for this worker's chunk.
 * @param initialBoundSq Initial shared bound used to seed local state.
 * @param globalBestDistSq Shared atomic global best squared distance.
 * @param outResult Output per-worker result.
 */
void parallel_worker_x_forward(
    const std::vector<Point>& px,
    std::size_t beginIndex,
    std::size_t endIndex,
    double initialBoundSq,
    std::atomic<double>& globalBestDistSq,
    ClosestPairResult& outResult)
{
    ClosestPairResult local;
    local.bestDistSq = initialBoundSq;

    if (px.size() < 2 || beginIndex >= endIndex) {
        outResult = local;
        return;
    }

    for (std::size_t i = beginIndex; i < endIndex; ++i) {
        const double sharedBoundSq = globalBestDistSq.load(std::memory_order_relaxed);
        if (sharedBoundSq < local.bestDistSq) {
            local.bestDistSq = sharedBoundSq;
        }

        const Point& p = px[i];

        std::size_t j = i + 1;
        while (j < px.size()) {
            const double dx = px[j].x - p.x;
            if ((dx * dx) >= local.bestDistSq) {
                break;
            }

            const Point& q = px[j];
            const double dy = q.y - p.y;

            if ((dy * dy) < local.bestDistSq) {
                ++local.comparisons;
                const double dSq = distance_squared_between(p, q);
                if (dSq < local.bestDistSq) {
                    local.bestDistSq = dSq;
                    local.a = p;
                    local.b = q;
                    publish_global_min(globalBestDistSq, dSq);
                }
            }

            ++j;

            const double refreshedBoundSq = globalBestDistSq.load(std::memory_order_relaxed);
            if (refreshedBoundSq < local.bestDistSq) {
                local.bestDistSq = refreshedBoundSq;
            }
        }
    }

    outResult = local;
}

/**
 * @brief Parallel CRNS closest-pair solver with chunked outer-loop partitioning.
 *
 * @param points Input points.
 * @param threadCount Number of worker threads.
 * @return ClosestPairResult Closest-pair result computed by parallel CRNS.
 */
ClosestPairResult chunked_parallel_cross_rank_search(
    const std::vector<Point>& points,
    std::size_t threadCount)
{
    ClosestPairResult finalResult;

    if (points.size() < 2) {
        return finalResult;
    }

    if (threadCount == 0) {
        threadCount = 1;
    }

    std::vector<Point> px = points;
    std::vector<Point> py = points;

    std::sort(px.begin(), px.end(), [](const Point& a, const Point& b) {
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.id < b.id;
    });

    std::sort(py.begin(), py.end(), [](const Point& a, const Point& b) {
        if (a.y != b.y)
            return a.y < b.y;
        if (a.x != b.x)
            return a.x < b.x;
        return a.id < b.id;
    });

    // Bootstrapping phase from adjacent pairs in both sorted orders.
    ClosestPairResult bootstrap = bootstrap_initial_bound(px, py);
    std::atomic<double> globalBestDistSq(bootstrap.bestDistSq);

    // Partition the outer loop across threads.
    const std::size_t n = px.size();
    if (threadCount > n) {
        threadCount = n;
    }

    std::vector<std::thread> threads;
    std::vector<ClosestPairResult> localResults(threadCount);

    threads.reserve(threadCount);

    const std::size_t baseChunk = n / threadCount;
    const std::size_t remainder = n % threadCount;

    std::size_t start = 0;
    for (std::size_t t = 0; t < threadCount; ++t) {
        const std::size_t chunkSize = baseChunk + (t < remainder ? 1 : 0);
        const std::size_t end = start + chunkSize;

        threads.emplace_back(
            parallel_worker_x_forward,
            std::cref(px),
            start,
            end,
            bootstrap.bestDistSq,
            std::ref(globalBestDistSq),
            std::ref(localResults[t]));

        start = end;
    }

    for (auto& th : threads) {
        th.join();
    }

    // Final reduction.
    finalResult = bootstrap;

    for (const auto& local : localResults) {
        finalResult.comparisons += local.comparisons;

        if (local.bestDistSq < finalResult.bestDistSq) {
            finalResult.bestDistSq = local.bestDistSq;
            finalResult.a = local.a;
            finalResult.b = local.b;
        }
    }

    return finalResult;
}

/**
 * @brief Prints a formatted closest-pair result block.
 *
 * @param label Section label to print above the result.
 * @param result Closest-pair result to display.
 */
void print_result(const std::string& label, const ClosestPairResult& result)
{
    std::cout << label << "\n";
    std::cout << "  Best distance: " << std::setprecision(15) << std::sqrt(result.bestDistSq) << "\n";
    std::cout << "  Point A: id=" << result.a.id
              << ", x=" << result.a.x
              << ", y=" << result.a.y << "\n";
    std::cout << "  Point B: id=" << result.b.id
              << ", x=" << result.b.x
              << ", y=" << result.b.y << "\n";
    std::cout << "  Distance comparisons: " << result.comparisons << "\n";
    std::cout << "\n";
}

/**
 * @brief Prints command-line usage and option descriptions.
 *
 * @param programName Executable name from argv[0].
 * @param maxThreads Maximum allowed thread count (hardware_concurrency).
 */
void print_usage(const char* programName, std::size_t maxThreads)
{
    std::cout << "Usage: " << programName
              << " [--points N | --points=N | -p N]"
              << " [--threads T | --threads=T | -t T]"
              << " [--distribution D | --distribution=D | -d D]"
              << " [--no-bruteforce]"
              << " [--run-divide-conquer]"
              << " [--run-hybrid]\n";
    std::cout << "  --points, -p   Number of random points to generate (positive integer)\n";
    std::cout << "  --threads, -t  Number of worker threads (positive integer, max "
              << maxThreads << ")\n";
    std::cout << "  --distribution, -d  Point distribution modifier in [0, 10]\n";
    std::cout << "                    10 = very even, 0 = very clumpy\n";
    std::cout << "  --no-bruteforce  Skip brute-force validation and timing\n";
    std::cout << "  --run-divide-conquer  Run serial and parallel divide-and-conquer variants\n";
    std::cout << "  --run-hybrid  Run hybrid parallel algorithm (D&C outer + CRNS leaves)\n";
}

/**
 * @brief Parses a strictly positive integer into std::size_t.
 *
 * @param value Input string.
 * @param out Parsed value on success.
 * @return bool True if parsing succeeded and value is in range.
 */
bool parse_positive_size_t(const std::string& value, std::size_t& out)
{
    try {
        std::size_t parsedChars = 0;
        const unsigned long long parsed = std::stoull(value, &parsedChars, 10);

        if (parsedChars != value.size()) {
            return false;
        }

        if (parsed == 0 || parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }

        out = static_cast<std::size_t>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/**
 * @brief Parses the distribution modifier in [0, 10].
 *
 * @param value Input string.
 * @param out Parsed distribution value on success.
 * @return bool True if parsing succeeded and value is finite/in-range.
 */
bool parse_distribution_modifier(const std::string& value, double& out)
{
    try {
        std::size_t parsedChars = 0;
        const double parsed = std::stod(value, &parsedChars);

        if (parsedChars != value.size() || !std::isfinite(parsed)) {
            return false;
        }

        if (parsed < 0.0 || parsed > 10.0) {
            return false;
        }

        out = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/**
 * @brief Program entry point: parses CLI flags, runs selected algorithms, and prints timings/results.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int Exit status code (0 on success, non-zero on argument errors).
 */
int main(int argc, char* argv[])
{
    std::size_t pointCount = 1000;
    const std::size_t hwThreadCount = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    std::size_t threadCount = hwThreadCount;
    double distributionModifier = 10.0;
    bool runBruteForceCheck = true;
    bool runDivideConquer = false;
    bool runHybrid = false;
    const unsigned int seed = 12345;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0], hwThreadCount);
            return 0;
        }

        if (arg == "--points" || arg == "-p") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            const std::string value = argv[++i];
            if (!parse_positive_size_t(value, pointCount)) {
                std::cerr << "Invalid point count: " << value << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            continue;
        }

        if (arg.rfind("--points=", 0) == 0) {
            const std::string value = arg.substr(std::string("--points=").size());
            if (!parse_positive_size_t(value, pointCount)) {
                std::cerr << "Invalid point count: " << value << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            continue;
        }

        if (arg == "--threads" || arg == "-t") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            const std::string value = argv[++i];
            std::size_t requestedThreads = 0;
            if (!parse_positive_size_t(value, requestedThreads)) {
                std::cerr << "Invalid thread count: " << value << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            if (requestedThreads > hwThreadCount) {
                std::cerr << "Requested thread count " << requestedThreads
                          << " exceeds hardware_concurrency() (" << hwThreadCount
                          << "). Using " << hwThreadCount << ".\n";
                threadCount = hwThreadCount;
            } else {
                threadCount = requestedThreads;
            }

            continue;
        }

        if (arg.rfind("--threads=", 0) == 0) {
            const std::string value = arg.substr(std::string("--threads=").size());
            std::size_t requestedThreads = 0;
            if (!parse_positive_size_t(value, requestedThreads)) {
                std::cerr << "Invalid thread count: " << value << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            if (requestedThreads > hwThreadCount) {
                std::cerr << "Requested thread count " << requestedThreads
                          << " exceeds hardware_concurrency() (" << hwThreadCount
                          << "). Using " << hwThreadCount << ".\n";
                threadCount = hwThreadCount;
            } else {
                threadCount = requestedThreads;
            }

            continue;
        }

        if (arg == "--distribution" || arg == "-d") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            const std::string value = argv[++i];
            if (!parse_distribution_modifier(value, distributionModifier)) {
                std::cerr << "Invalid distribution modifier: " << value
                          << " (expected number in [0, 10])\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            continue;
        }

        if (arg.rfind("--distribution=", 0) == 0) {
            const std::string value = arg.substr(std::string("--distribution=").size());
            if (!parse_distribution_modifier(value, distributionModifier)) {
                std::cerr << "Invalid distribution modifier: " << value
                          << " (expected number in [0, 10])\n";
                print_usage(argv[0], hwThreadCount);
                return 1;
            }

            continue;
        }

        if (arg == "--no-bruteforce") {
            runBruteForceCheck = false;
            continue;
        }

        if (arg == "--run-divide-conquer") {
            runDivideConquer = true;
            continue;
        }

        if (arg == "--run-hybrid") {
            runHybrid = true;
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0], hwThreadCount);
        return 1;
    }

    std::vector<Point> points = generate_points(pointCount, seed, distributionModifier);

    const auto t1 = std::chrono::high_resolution_clock::now();
    ClosestPairResult customSerialResult = cross_rank_serial_search(points);
    const auto t2 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> customSerialMs = t2 - t1;

    const auto t3 = std::chrono::high_resolution_clock::now();
    ClosestPairResult customParallelResult = chunked_parallel_cross_rank_search(points, threadCount);
    const auto t4 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> customParallelMs = t4 - t3;

    ClosestPairResult divideConquerSerialResult;
    ClosestPairResult divideConquerParallelResult;
    ClosestPairResult hybridResult;
    std::chrono::duration<double, std::milli> divideConquerSerialMs(0.0);
    std::chrono::duration<double, std::milli> divideConquerParallelMs(0.0);
    std::chrono::duration<double, std::milli> hybridMs(0.0);

    if (runDivideConquer) {
        const auto td1 = std::chrono::high_resolution_clock::now();
        divideConquerSerialResult = divide_and_conquer_serial_closest_pair(points);
        const auto td2 = std::chrono::high_resolution_clock::now();
        divideConquerSerialMs = td2 - td1;

        const auto td3 = std::chrono::high_resolution_clock::now();
        divideConquerParallelResult = divide_and_conquer_parallel_closest_pair(points, threadCount);
        const auto td4 = std::chrono::high_resolution_clock::now();
        divideConquerParallelMs = td4 - td3;
    }

    if (runHybrid) {
        const auto th1 = std::chrono::high_resolution_clock::now();
        hybridResult = hybrid_parallel_dnc_crns_closest_pair(points, threadCount);
        const auto th2 = std::chrono::high_resolution_clock::now();
        hybridMs = th2 - th1;
    }

    ClosestPairResult bruteForceResult;
    std::chrono::duration<double, std::milli> bruteMs(0.0);

    if (runBruteForceCheck) {
        const auto tb1 = std::chrono::high_resolution_clock::now();
        bruteForceResult = brute_force_closest_pair(points);
        const auto tb2 = std::chrono::high_resolution_clock::now();
        bruteMs = tb2 - tb1;
    }

    print_result("CRNS (Serial):", customSerialResult);
    print_result(
        "CRNS (Parallel Shared-Bound):",
        customParallelResult);

    if (runDivideConquer) {
        print_result("Divide and Conquer (Serial):", divideConquerSerialResult);
        print_result("Divide and Conquer (Parallel):", divideConquerParallelResult);
    }
    if (runHybrid) {
        print_result("Hybrid (Parallel D&C + CRNS Leaves):", hybridResult);
    }

    if (runBruteForceCheck) {
        print_result("Brute Force:", bruteForceResult);
    }

    std::cout << "Thread count: " << threadCount << "\n";
    std::cout << "Point count: " << pointCount << "\n";
    std::cout << "Distribution modifier: " << distributionModifier << "\n";

    if (runBruteForceCheck) {
        const double scale = std::max(
            1.0,
            std::max(
                std::max(customSerialResult.bestDistSq, customParallelResult.bestDistSq),
                runDivideConquer
                    ? std::max(divideConquerParallelResult.bestDistSq, bruteForceResult.bestDistSq)
                    : bruteForceResult.bestDistSq));

        const bool customSerialMatch = std::abs(customSerialResult.bestDistSq - bruteForceResult.bestDistSq) <= (1e-12 * scale);
        const bool customParallelMatch = std::abs(customParallelResult.bestDistSq - bruteForceResult.bestDistSq) <= (1e-12 * scale);

        std::cout << "CRNS serial matches brute force: "
                  << (customSerialMatch ? "true" : "false") << "\n";
        std::cout << "CRNS parallel matches brute force: "
                  << (customParallelMatch ? "true" : "false") << "\n";

        if (runDivideConquer) {
            const bool divideConquerSerialMatch = std::abs(divideConquerSerialResult.bestDistSq - bruteForceResult.bestDistSq) <= (1e-12 * scale);
            const bool divideConquerParallelMatch = std::abs(divideConquerParallelResult.bestDistSq - bruteForceResult.bestDistSq) <= (1e-12 * scale);

            std::cout << "Divide-and-conquer serial matches brute force: "
                      << (divideConquerSerialMatch ? "true" : "false") << "\n";
            std::cout << "Divide-and-conquer parallel matches brute force: "
                      << (divideConquerParallelMatch ? "true" : "false") << "\n";
        }

        if (runHybrid) {
            const bool hybridMatch = std::abs(hybridResult.bestDistSq - bruteForceResult.bestDistSq) <= (1e-12 * scale);
            std::cout << "Hybrid matches brute force: " << (hybridMatch ? "true" : "false")
                      << "\n";
        }

        if (customSerialResult.comparisons > 0) {
            std::cout << "Comparison reduction factor (brute / CRNS serial): "
                      << static_cast<double>(bruteForceResult.comparisons) / static_cast<double>(customSerialResult.comparisons)
                      << "\n";
        }

        if (customParallelResult.comparisons > 0) {
            std::cout << "Comparison reduction factor (brute / CRNS parallel): "
                      << static_cast<double>(bruteForceResult.comparisons) / static_cast<double>(customParallelResult.comparisons)
                      << "\n";
        }
    } else {
        std::cout << "Brute force check: disabled\n";

        if (runDivideConquer) {
            const double scale = std::max(
                1.0,
                std::max(
                    std::max(customSerialResult.bestDistSq, customParallelResult.bestDistSq),
                    std::max(
                        divideConquerSerialResult.bestDistSq,
                        divideConquerParallelResult.bestDistSq)));

            const bool customSerialVsDivideConquerSerial = std::abs(customSerialResult.bestDistSq - divideConquerSerialResult.bestDistSq) <= (1e-12 * scale);
            const bool customParallelVsDivideConquerParallel = std::abs(
                                                                   customParallelResult.bestDistSq - divideConquerParallelResult.bestDistSq)
                <= (1e-12 * scale);

            std::cout << "CRNS serial matches divide-and-conquer serial: "
                      << (customSerialVsDivideConquerSerial ? "true" : "false") << "\n";
            std::cout << "CRNS parallel matches divide-and-conquer parallel: "
                      << (customParallelVsDivideConquerParallel ? "true" : "false") << "\n";
        }

        if (runHybrid) {
            const double scale = std::max(
                1.0,
                std::max(
                    hybridResult.bestDistSq,
                    std::max(customSerialResult.bestDistSq, customParallelResult.bestDistSq)));
            const bool hybridVsCrnsParallel = std::abs(hybridResult.bestDistSq - customParallelResult.bestDistSq) <= (1e-12 * scale);
            std::cout << "Hybrid matches CRNS parallel: "
                      << (hybridVsCrnsParallel ? "true" : "false") << "\n";
        }
    }

    std::cout << "CRNS serial time (ms): " << customSerialMs.count() << "\n";
    std::cout << "CRNS parallel time (ms): " << customParallelMs.count() << "\n";
    if (runDivideConquer) {
        std::cout << "Divide-and-conquer serial time (ms): " << divideConquerSerialMs.count()
                  << "\n";
        std::cout << "Divide-and-conquer parallel time (ms): " << divideConquerParallelMs.count()
                  << "\n";
    }
    if (runHybrid) {
        std::cout << "Hybrid time (ms): " << hybridMs.count() << "\n";
    }
    if (runBruteForceCheck) {
        std::cout << "Brute force time (ms): " << bruteMs.count() << "\n";
    }

    return 0;
}
