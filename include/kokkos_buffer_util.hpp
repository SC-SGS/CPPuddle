#include <Kokkos_Core.hpp>

template <class kokkos_type>
class weak_recycled_view;

template <class kokkos_type, class alloc_type, class element_type>
class recycled_view : public kokkos_type
{
private:
    static alloc_type allocator;
    size_t total_elements;

public:
    template <class... Args>
    recycled_view(Args... args) : kokkos_type(allocator.allocate(kokkos_type::required_allocation_size(args...) / sizeof(element_type)), args...),
                                  total_elements(kokkos_type::required_allocation_size(args...) / sizeof(element_type))
    {
        //std::cout << "Got buffer for " << total_elements << std::endl;
        std::cout << "Creating view" << std::endl;
    }

    recycled_view(const recycled_view<kokkos_type, alloc_type, element_type> &other) : kokkos_type(other)
    {
        total_elements = other.total_elements;
        // std::cerr << "copy" << std::endl;

        std::cout << "Copying view2" << std::endl;
        allocator.increase_usage_counter(this->data(), this->total_elements);

    }

    recycled_view<kokkos_type, alloc_type, element_type> &operator=(const recycled_view<kokkos_type, alloc_type, element_type> &other)
    {
        allocator.deallocate(this->data(), total_elements);
        kokkos_type::operator=(other);
        total_elements = other.total_elements;
        allocator.increase_usage_counter(other.data(), other.total_elements);
        std::cout << "Assignement view" << std::endl;
        return *this;
    }

    recycled_view(recycled_view<kokkos_type, alloc_type, element_type> &&other) : kokkos_type(other)
    {
        total_elements = other.total_elements;
        allocator.increase_usage_counter(other.data(), other.total_elements);
        std::cout << "Moving view" << std::endl;
    }

    recycled_view<kokkos_type, alloc_type, element_type> &operator=(recycled_view<kokkos_type, alloc_type, element_type> &&other)
    {
        allocator.deallocate(this->data(), total_elements);
        kokkos_type::operator=(other);
        total_elements = other.total_elements;
        allocator.increase_usage_counter(other.data(), other.total_elements);
        std::cout << "Moving view assign" << std::endl;
        return *this;
    }

    virtual ~recycled_view(void)
    {
        std::cout << "Dellocating view" << std::endl;
        allocator.deallocate(this->data(), total_elements);
    }

    // get a view that does not increase the reference count -- lifetime of parent has to be guaranteed by user
    weak_recycled_view<kokkos_type> weak() const
    {
        return weak_recycled_view<kokkos_type>(*this);
    }
};

template <class kokkos_type, class alloc_type, class element_type>
alloc_type recycled_view<kokkos_type, alloc_type, element_type>::allocator;

template <class kokkos_type>
class weak_recycled_view : public kokkos_type
{
private:
public:
    ~weak_recycled_view(void)
    {
    }

    template <class... Args>
    weak_recycled_view(const recycled_view<Args...> &other) : kokkos_type(other)
    {
    }

    template <class... Args>
    weak_recycled_view &operator=(const recycled_view<Args...> &other)
    {
        this = weak_recycled_view(other);
    }

    explicit weak_recycled_view() = delete;
    weak_recycled_view(const weak_recycled_view &other) = default;
    weak_recycled_view(weak_recycled_view &&other) noexcept = default;
    weak_recycled_view &operator=(const weak_recycled_view &other) = default;
    weak_recycled_view &operator=(weak_recycled_view &&other) noexcept = default;
};


/**
 * get an MDRangePolicy suitable for iterating the views
 * 
 * @param executor          a kokkos ExecutionSpace, e.g. hpx::kokkos::make_execution_space<Kokkos::Cuda>()
 * @param view_to_iterate   the view that needs to be iterated
 */
template <typename Executor, typename ViewType>
auto get_iteration_policy(const Executor& executor, const ViewType& view_to_iterate)
{
    constexpr auto rank = ViewType::ViewTraits::rank;
    const Kokkos::Array<int64_t, rank> zeros{};
    Kokkos::Array<int64_t, rank> extents;
    for (int i = 0; i < rank; ++i)
    {
        extents[i] = view_to_iterate.extent(i);
    }

  // TODO what exactly does HintLightWeight do? cf. https://github.com/kokkos/kokkos/issues/1723
  return Kokkos::Experimental::require(Kokkos::MDRangePolicy<Executor, Kokkos::Rank<rank>>(executor,
                                                                                           zeros, extents),
                                       Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  // return Kokkos::MDRangePolicy<Executor, Kokkos::Rank<rank>>(executor, zeros, extents);
}
