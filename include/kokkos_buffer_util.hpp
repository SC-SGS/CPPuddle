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
    }

    recycled_view(const recycled_view<kokkos_type, alloc_type, element_type> &other) : kokkos_type(other)
    {
        total_elements = other.total_elements;
        // std::cerr << "copy" << std::endl;
        allocator.increase_usage_counter(other.data(), other.total_elements);
    }

    recycled_view<kokkos_type, alloc_type, element_type> &operator=(const recycled_view<kokkos_type, alloc_type, element_type> &other)
    {
        kokkos_type::operator=(other);
        total_elements = other.total_elements;
        allocator.increase_usage_counter(other.data(), other.total_elements);
        return *this;
    }

    recycled_view(recycled_view<kokkos_type, alloc_type, element_type> &&other) : kokkos_type(other)
    {
        total_elements = other.total_elements;
        // so that is doesn't matter if deallocate is called in the moved-from object
        allocator.increase_usage_counter(other.data(), other.total_elements);
    }

    recycled_view<kokkos_type, alloc_type, element_type> &operator=(recycled_view<kokkos_type, alloc_type, element_type> &&other)
    {
        kokkos_type::operator=(other);
        total_elements = other.total_elements;
        // so that is doesn't matter if deallocate is called in the moved-from object
        allocator.increase_usage_counter(other.data(), other.total_elements);
        return *this;
    }

    ~recycled_view(void)
    {
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
