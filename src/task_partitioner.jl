"""
    $(TYPEDEF)

Partitioning `num_elems` elements / jobs over maximum `max_tasks` tasks with minimum `min_elems`
elements per task.

# Methods
    TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)

# Fields
    $(TYPEDFIELDS)

# Examples

```jldoctest
using ImplicitBVH: TaskPartitioner

# Divide 10 elements between 4 tasks
tp = TaskPartitioner(10, 4)
for i in 1:tp.num_tasks
    @show tp[i]
end

# output
tp[i] = 1:3
tp[i] = 4:6
tp[i] = 7:9
tp[i] = 10:10
```

```jldoctest
using ImplicitBVH: TaskPartitioner

# Divide 20 elements between 6 tasks with minimum 5 elements per task.
# Not all tasks will be required
tp = TaskPartitioner(20, 6, 5)
for i in 1:tp.num_tasks
    @show tp[i]
end

# output
tp[i] = 1:5
tp[i] = 6:10
tp[i] = 11:15
tp[i] = 16:20
```
"""
struct TaskPartitioner
    num_elems::Int
    max_tasks::Int
    min_elems::Int

    # Computed
    num_tasks::Int
    task_istarts::Vector{Int}

    # Full inner constructor
    function TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, task_istarts)
        new(num_elems, max_tasks, min_elems, num_tasks, task_istarts)
    end

    # Incomplete constructor, not defining the task_istarts vector in case of single task
    function TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks)
        num_tasks == 1 || throw(ArgumentError("incomplete constructor is only for num_tasks == 1"))
        new(num_elems, max_tasks, min_elems, 1)
    end
end


function TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    # Simple correctness checks
    @argcheck num_elems >= 0
    @argcheck max_tasks > 0
    @argcheck min_elems > 0

    # Number of tasks needed to have at least `min_elems` per task
    num_tasks = min(max_tasks, num_elems ÷ min_elems)
    if num_tasks <= 1
        num_tasks = 1
        return TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks)
    end

    # Each task gets at least (num_elems ÷ num_tasks) elements; the remaining are redistributed
    # among the first (num_elems % num_tasks) tasks, i.e. they get one extra element
    per_task, remaining = divrem(num_elems, num_tasks)

    # Store starting index of each task
    task_istarts = Vector{Int}(undef, num_tasks)
    istart = 1
    @inbounds for i in 1:num_tasks
        task_istarts[i] = istart
        istart += i <= remaining ? per_task + 1 : per_task
    end

    TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks, task_istarts)
end


function Base.getindex(tp::TaskPartitioner, itask::Integer)

    @boundscheck 1 <= itask <= tp.num_tasks || throw(BoundsError(tp, itask))

    # Special-cased for single task, in which case tp.task_istarts was not defined / allocated
    if tp.num_tasks == 1
        return 1:tp.num_elems
    end

    task_istart = @inbounds tp.task_istarts[itask]
    if itask == tp.num_tasks
        return task_istart:tp.num_elems
    else
        task_istop = @inbounds tp.task_istarts[itask + 1] - 1
        return task_istart:task_istop
    end
end


Base.firstindex(tp::TaskPartitioner) = 1
Base.lastindex(tp::TaskPartitioner) = tp.num_tasks
Base.length(tp::TaskPartitioner) = tp.num_tasks




"""
Partition `num_elems` jobs across at most `num_tasks` parallel tasks with at least `min_elems` per
task, calling `f(start_index:end_index)`, where the indices are between 1 and `num_elems`.

# Examples
A toy example showing outputs:
```julia
num_elems = 4
task_partition(println, num_elems)

# Output, possibly in a different order due to threading order
1:1
4:4
2:2
3:3
```

This function is probably most useful with a do-block, e.g.:
```julia
task_partition(4) do irange
    some_long_computation(param1, param2, irange)
end
```
"""
function task_partition(f, num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    num_elems >= 0 || throw(ArgumentError("num_elems must be >= 0"))
    max_tasks > 0 || throw(ArgumentError("max_tasks must be > 0"))
    min_elems > 0 || throw(ArgumentError("min_elems must be > 0"))

    if min(max_tasks, num_elems ÷ min_elems) <= 1
        @inline f(1:num_elems)
    else
        # Compiler should decide if this should be inlined; threading adds quite a bit of code, it
        # is faster (as seen in Cthulhu) to keep it in a separate self-contained function
        _task_partition_threads(f, num_elems, max_tasks, min_elems)
    end
    nothing
end


function _task_partition_threads(f, num_elems, max_tasks, min_elems)
    tp = TaskPartitioner(num_elems, max_tasks, min_elems)
    tasks = Vector{Task}(undef, tp.num_tasks - 1)

    # Launch first N - 1 tasks
    for i in 1:tp.num_tasks - 1
        tasks[i] = Threads.@spawn f(@inbounds(tp[i]))
    end

    # Execute task N on this main thread
    f(@inbounds(tp[tp.num_tasks]))

    # Wait for the tasks to finish
    @inbounds for i in 1:tp.num_tasks - 1
        wait(tasks[i])
    end
end

