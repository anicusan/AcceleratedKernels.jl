function unrolled_map_index(f, tuple_vector::Tuple)
    unrolled_map(FixedRange{1, length(tuple_vector)}()) do i
        @inline f(i)
    end
end


@inline @unroll function unrolled_foreach_index(f, tuple_vector::Tuple)
    @unroll for i in 1:length(tuple_vector)
        @inline f(tuple_vector, i)
    end
    nothing
end
