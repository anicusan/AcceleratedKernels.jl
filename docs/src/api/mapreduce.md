### MapReduce

Equivalent to `reduce(op, map(f, iterable))`, without saving the intermediate mapped collection; can be used to e.g. split documents into words (map) and count the frequency thereof (reduce).
- **Other names**: `transform_reduce`, some `fold` implementations include the mapping function too.

**New in AcceleratedKernels 0.2.0: N-dimensional reductions via the `dims` keyword**

---

```@docs
AcceleratedKernels.mapreduce
```
