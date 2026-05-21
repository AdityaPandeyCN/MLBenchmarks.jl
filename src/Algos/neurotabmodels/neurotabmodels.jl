using NeuroTabModels

_as_iter(x) = x isa Union{AbstractArray,AbstractRange,Tuple} ? x : (x,)

include("neurotrees.jl")
include("tabM.jl")
include("modernnca.jl")

_neurotab_model_type(hyper) = replace(string(get(hyper, :arch_name, "NeuroTabModels")), "Config" => "")

function _as_metric_vector(p)
    p isa AbstractMatrix && size(p, 2) == 1 && return p[:, 1]
    return p
end

function run_experiment(
    ::Val{:NeuroTabModels},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    print_every_n=10
)
    dtrain = data[:dtrain]
    deval = data[:deval]
    dtest = data[:dtest]
    feature_names = data[:feature_names]
    target_name = data[:target_name]

    results = OrderedDict{Symbol,Any}[]

    # warmup
    hyper = copy(first(hyper_list))
    hyper[:nrounds] = 1
    config = NeuroTabModels.NeuroTabRegressor(; hyper...)
    NeuroTabModels.fit(config, dtrain; deval, feature_names, target_name)

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(NeuroTabModels) loop $i"
        config = NeuroTabModels.NeuroTabRegressor(; hyper...)
        train_time = @elapsed m = NeuroTabModels.fit(
            config,
            dtrain;
            deval,
            feature_names,
            target_name,
            print_every_n=print_every_n
        )
        p_eval = m(deval)
        p_test = m(dtest)
        p_eval = _as_metric_vector(p_eval)
        p_test = _as_metric_vector(p_test)

        res = OrderedDict{Symbol,Any}(
            :model_type => _neurotab_model_type(hyper),
            :hyper_id => i,
            :train_time => train_time,
            :best_nround => m.info[:logger][:best_iter],
            :batchsize => get(hyper, :batchsize, missing),
            :epochs => get(hyper, :nrounds, missing),
            :seed => get(hyper, :seed, missing),
            :arch_config => string(get(hyper, :arch_config, missing)),
            :embedding_config => string(get(hyper, :embedding_config, missing)),
            :lr => get(hyper, :lr, missing),
            :wd => get(hyper, :wd, missing),
        )

        for metric in metrics
            fun = metric_dict[metric]
            res[Symbol("eval_", metric)] = fun(p_eval, deval[!, target_name])
            res[Symbol("test_", metric)] = fun(p_test, dtest[!, target_name])
        end
        push!(results, res)
    end

    return DataFrame(results)
end

