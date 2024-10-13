## Debugging Kernels

As the compilation pipeline of GPU kernels is different to that of base Julia, error messages also look different - for example, where Julia would insert an exception when a variable name was not defined (e.g. we had a typo), a GPU kernel throwing exceptions cannot be compiled and instead you'll see some cascading errors like `"[...] compiling [...] resulted in invalid LLVM IR"` caused by `"Reason: unsupported use of an undefined name"` resulting in `"Reason: unsupported dynamic function invocation"`, etc.

Thankfully, there are only about 3 types of such error messages and they're not that scary when you look into them.


### Undefined Variables / Typos

If you misspell a variable name, Julia would insert an exception:

```julia
function set_color(v, color)
    AK.foreachindex(v) do i
        v[i] = colour           # Grab your porridge
    end
end
```

However, exceptions cannot be compiled on GPUs and you will see cascading errors like below:

![Undefined Name Error](./assets/debug_undefined_name.png)

The key thing to look for is `undefined name`, then search for it in your code.


### Exceptions and Checks that `throw`

As mentioned above, exceptions cannot be compiled in GPU kernels; however, many normal-looking functions that we reference in kernels may contain argument-checking. If it cannot be proved that a check branch would not throw an exception, you will see a similar cascade of errors. For example, casting a `Float32` to an `Int32` includes an `InexactError` exception check - see this tame-looking code:

```julia
function mymul!(v)
    AK.foreachindex(v) do i
        v[i] *= 2f0
    end
end

v = MtlArray(1:1000)
mymul!(v)
```

See any problem with it? The `MtlArray(1:1000)` creates a GPU vector filled with `Int64` values, but within `foreachindex` we do `v[i] *= 2.0`. We are multiplying an `Int64` by a `Float32`, resulting in a `Float32` value that we try to write back into `v` - this may throw an exception, like in normal Julia code:

```julia
julia> x = [1, 2, 3];
julia> x[1] = 42.5
ERROR: InexactError: Int64(42.5)
```

On GPUs you will see an error like this:

![Check Exception Error](./assets/debug_check_exception.png)

Note the error stack: `setindex!`, `convert`, `Int64`, `box_float32` - because of the exception check, we have a type instability, which in turn results in boxing values behind pointers, in turn resulting in dynamic memory allocation and finally the error we see at the top, `unsupported call to gpu_malloc`.

You may need to do your correctness checks manually, without exceptions; in this specific case, if we did want to cast a `Float32` to an `Int`, we could use `unsafe_trunc(T, x)` - though be careful when using unsafe functions that you understand their behaviour and assumptions (e.g. `log` has a `DomainError` check for negative values):

```julia
function mymul!(v)
    AK.foreachindex(v) do i
        v[i] = unsafe_trunc(eltype(v), v[i] * 2.5f0)
    end
end

v = MtlArray(1:1000)
mymul!(v)
```


### Type Instability / Global Variables

Types must be known to be captured and compiled within GPU kernels. Global variables without `const` are *not* type-stable, as you could associate a different value later on in a script:

```julia
v = MtlArray(1:1000)

AK.foreachindex(v) do i
    v[i] *= 2
end

v = "potato"
```

The error stack is a bit more difficult here:

![Type Unstable Error](./assets/debug_type_unstable.png)

You see a few `dynamic function invocation`, an `unsupported call to gpu_malloc`, and a bit further down a `box`. The more operations you do on the type-unstable object, the more `dynamic function invocation` errors you'll see. These would also be the steps Base Julia would take to allow dynamically-changing objects: they'd be put in a `Box` behind pointers, and allocated on the heap. In a way, it is better that we cannot do that on a GPU, as it hurts performance massively.

There are two ways to solve this - if you really want to use global variables in a script, put them behind a `const`:

```julia
const v = MtlArray(1:1000)

AK.foreachindex(v) do i
    v[i] *= 2
end

# This would give you an error now
# v = "potato"
```

Or better, use functions:

```julia
function mymul!(v, x)
    AK.foreachindex(v) do i
        v[i] *= x
    end
end

v = MtlArray(1:1000)
mymul!(v, 2)
```

Note that Julia's lambda capture is very powerful - inside `AK.foreachindex` you can references other objects from within the function (like `x`), without explicitly passing them to the GPU.


### Apple Metal Only: Float64 is not Supported

Mac GPUs do not natively support `Float64` values; there is a high-level check when trying to create an array:

```julia
julia> x = MtlArray([1.0, 2.0, 3.0])
ERROR: Metal does not support Float64 values, try using Float32 instead
```

However, if we tried to use / convert values in a kernel to a `Float64`:

```julia
function mymul!(v, x)
    AK.foreachindex(v) do i
        v[i] *= x
    end
end

v = MtlArray{Float32}(1:1000)
mymul!(v, 2.0)
```

Note that we try to multiply `Float32` values by `2.0`, which is a `Float64` - in which case we get:

```bash
ERROR: LoadError: Compilation to native code failed; see below for details.
[...]
caused by: NSError: Compiler encountered an internal error (AGXMetalG15X_M1, code 3)
[...]
```

Change the `2.0` to `2.0f0` or `Float32(2)`; in kernels with generic types (that are supposed to work on multiple possible input types), do use the same types as your inputs, using e.g. `T = eltype(v)` then `zero(T)`, `T(42)`, etc.


---

For other library-related problems, feel free to post a GitHub issue. For help implementing new code, or just advice, you can also use the [Julia Discourse](https://discourse.julialang.org/c/domain/gpu/11) forum, the community is incredibly helpful.

