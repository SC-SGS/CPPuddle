/// \file Contains an abstraction for Kokkos views intended for use with CUDA stream synchronization

#ifndef HPX_KOKKOS_VIEWPOOL_HPP
#define HPX_KOKKOS_VIEWPOOL_HPP

#include <Kokkos_Core.hpp>
#include <any>
#include <boost/any.hpp>

namespace hpx
{

namespace kokkos
{

/** view_bundle
 * A bundle of Kokkos views representing the same data, mirrored between 
 * CUDA and CUDA pinned host memory. 
 */
template <class DataType>
struct view_bundle
{
    //TODO allow passing different memory spaces than CUDA+pinned (=current default)
    view_bundle(const std::string label = "bundle")
    {
        device_view_ = Kokkos::View<DataType, Kokkos::CudaSpace>(Kokkos::ViewAllocateWithoutInitializing(label));
        host_view_ = Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), device_view_);
    }

    // view_bundle(const view_bundle& other) = delete;
    // view_bundle& operator=(const view_bundle& other) = delete;

    /**
     * a deep copy that returns a future (realized by callback on the executor's stream)
     * only wait for the last future to get a series of operations that are to be stream-synchronized
     * 
     * @param executor an hpx executor instance or kokkos ExecutionSpace, e.g. hpx::kokkos::make_execution_space<Kokkos::Cuda>()
     * @see hpx::kokkos::deep_copy_async 
     */
    template <typename Executor,
              typename MemorySpaceTo = Kokkos::CudaHostPinnedSpace::memory_space,
              typename MemorySpaceFrom = Kokkos::Cuda::memory_space>
    //TODO not sure how/if Kokkos distinguishes memory spaces of the same type (multiple devices) => maybe remove parameters
    hpx::future<void> async_deep_copy(Executor executor, MemorySpaceTo memory_space_to, MemorySpaceFrom memory_space_from);

    /**
     * synchronous blocking deep_copy (on the executor's stream)
     * 
     * ...same as current Kokkos implementhpx::kokkos::ation
     */
    template <typename Executor,
              typename MemorySpaceTo = Kokkos::Cuda::memory_space,
              typename MemorySpaceFrom = Kokkos::CudaHostPinnedSpace::memory_space>
    void sync_deep_copy(Executor executor, MemorySpaceTo memory_space_to, MemorySpaceFrom memory_space_from);

    Kokkos::View<DataType, Kokkos::CudaSpace> device_view_;
    decltype(Kokkos::create_mirror_view(Kokkos::CudaHostPinnedSpace(), device_view_)) host_view_;
};

/** view_pool 
 * in analogy to a thread pool, the view pool manages limited 
 * view/buffer resources that can be requested at runtime. 
 * Here, these are view_bundles
 */
struct view_pool
{
    // TODO: can this be more generic? Should it know about memory spaces?
    view_pool() : view_bundles_{}
    {
    }

    /**
     * request reference to a view bundle; call will block until available
     * @tparam DataType the compile time data type passed to View construction 
     *         (containing the extents in all dimensions)
     * TODO maybe return weak pointer
     */
    template <class DataType>
    view_bundle<DataType> request()
    {
        view_bundles_.emplace_back();
        // return boost::any_cast<view_bundle<DataType>>(view_bundles_.back());
        //TODO implement properly
        return view_bundles_.back();
    }

    template <class DataType>
    void release(const view_bundle<DataType> &bundle){}

    //TODO what is the nicest way of managing the buffers?
    // thread local: array of bundles and lookup table
    // locality local: mutexes?
    // something between: allocate all dimensions and views in the beginning, or
    // allocate everything as required (but never free ;) )

    // std::vector<view_bundle<boost::any>> view_bundles_;
    std::vector<view_bundle<double[50][10]>> view_bundles_;
    // std::vector<view_bundle<std::any>> view_bundles_; // error: namespace "std" has no member "any"

};

/**
 * RAII guard to safely get and release a view_bundle
 */
template <class DataType>
struct bundle_guard
{
    bundle_guard(view_pool &viewpool) : viewpool_{viewpool}, view_bundle_{viewpool.request<DataType>()}
    {
    }

    ~bundle_guard()
    {
        viewpool_.release<DataType>(view_bundle_);
    }

    view_bundle<DataType> &get()
    {
        return view_bundle_;
    }

private:
    view_pool &viewpool_;
    view_bundle<DataType> view_bundle_; //TODO make a reference?
};

} // namespace kokkos

} // namespace hpx

#endif