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
using AcceleratedKernels: TaskPartitioner

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
    num_tasks::Int      # computed
end


function TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    # Number of tasks needed to have at least `min_nodes` per task
    num_tasks = num_elems ÷ max_tasks >= min_elems ? max_tasks : num_elems ÷ min_elems
    if num_tasks < 1
        num_tasks = 1
    end

    TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks)
end


function Base.getindex(tp::TaskPartitioner, itask::Integer)

    @boundscheck 1 <= itask <= tp.num_tasks || throw(BoundsError(tp, itask))

    # Compute element indices handled by this task
    per_task = (tp.num_elems + tp.num_tasks - 1) ÷ tp.num_tasks

    task_istart = (itask - 1) * per_task + 1
    task_istop = min(itask * per_task, tp.num_elems)

    task_istart:task_istop
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
    tp = TaskPartitioner(num_elems, max_tasks, min_elems)
    if tp.num_tasks == 1
        f(1:num_elems)
    else
        tasks = Vector{Task}(undef, tp.num_tasks - 1)

        # Launch first N - 1 tasks
        @inbounds for i in 1:tp.num_tasks - 1
            tasks[i] = Threads.@spawn f(tp[i])
        end

        # Execute task N on this main thread
        @inbounds f(tp[tp.num_tasks])

        # Wait for the tasks to finish
        @inbounds for i in 1:tp.num_tasks - 1
            wait(tasks[i])
        end
    end
    nothing
end

