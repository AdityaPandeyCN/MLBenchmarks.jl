using DelimitedFiles

const _TALENT_RUNNER_DIR = @__DIR__
const _TALENT_RUNNER_SCRIPT = joinpath(_TALENT_RUNNER_DIR, "talent_runner.py")
const _TALENT_PYTHON = get(ENV, "MLBENCHMARKS_TALENT_PYTHON", "/usr/bin/python3")

_json(x::Nothing) = "null"
_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = string(x)
_json(x::Symbol) = _json(string(x))
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(x::AbstractVector) = "[" * join((_json(v) for v in x), ",") * "]"
_json(x::AbstractDict) = "{" * join((_json(string(k)) * ":" * _json(v) for (k, v) in x), ",") * "}"

function _matrix(df, feature_names)
    Matrix{Float32}(df[:, collect(feature_names)])
end

function _target(df, target_name)
    Vector{Float32}(df[!, target_name])
end

function _talent_model_type(hyper)
    get(hyper, :model_type, "talent")
end

function _write_json(path, value)
    open(path, "w") do io
        write(io, _json(value))
    end
end

function _read_key_values(path)
    values = Dict{String,String}()
    for line in eachline(path)
        key, value = split(line, "="; limit=2)
        values[key] = value
    end
    return values
end

function _run_talent_python(spec)
    _write_json(spec["spec_path"], spec)
    run(Cmd([_TALENT_PYTHON, _TALENT_RUNNER_SCRIPT, spec["spec_path"]]))
    return _read_key_values(spec["result_path"])
end

function get_hyper_talent_modernnca(
    hyper_size;
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    dim=128,
    d_block=256,
    n_blocks=2,
    dropout=0.1,
    temperature=1.0,
    sample_rate=0.01,
    batchsize=256,
    seed=123,
    normalization="standard",
    use_float=true,
)

    hyper_list = Dict{Symbol,Any}[]

    for _lr in _as_iter(lr), _wd in _as_iter(wd), _dim in _as_iter(dim), _d_block in _as_iter(d_block),
        _n_blocks in _as_iter(n_blocks), _dropout in _as_iter(dropout), _temperature in _as_iter(temperature),
        _sample_rate in _as_iter(sample_rate), _batchsize in _as_iter(batchsize), _seed in _as_iter(seed)

        config = Dict{String,Any}(
            "model" => Dict{String,Any}(
                "dim" => _dim,
                "dropout" => _dropout,
                "d_block" => _d_block,
                "n_blocks" => _n_blocks,
                "temperature" => _temperature,
                "sample_rate" => _sample_rate,
                "num_embeddings" => nothing,
            ),
            "training" => Dict{String,Any}(
                "lr" => _lr,
                "weight_decay" => _wd,
                "n_bins" => 2,
            ),
            "general" => Dict{String,Any}(),
        )

        push!(hyper_list, Dict{Symbol,Any}(
            :model_type => "modernNCA",
            :config => config,
            :cat_policy => "tabr_ohe",
            :num_policy => "none",
            :normalization => normalization,
            :nrounds => nrounds,
            :batchsize => _batchsize,
            :seed => _seed,
            :use_float => use_float,
            :arch_config => config["model"],
            :lr => _lr,
            :wd => _wd,
        ))
    end

    rng = Xoshiro(123)
    return sample(rng, hyper_list, hyper_size, replace=false)
end

function get_hyper_talent_tabm(
    hyper_size;
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    arch_type="tabm",
    k=16,
    d_embedding=16,
    n_frequencies=77,
    frequency_scale=0.04431360576139521,
    d_block=128,
    n_blocks=2,
    dropout=0.1,
    batchsize=256,
    seed=123,
    normalization="standard",
    use_float=true,
)

    hyper_list = Dict{Symbol,Any}[]

    for _lr in _as_iter(lr), _wd in _as_iter(wd), _k in _as_iter(k), _d_embedding in _as_iter(d_embedding),
        _d_block in _as_iter(d_block), _n_blocks in _as_iter(n_blocks), _dropout in _as_iter(dropout),
        _batchsize in _as_iter(batchsize), _seed in _as_iter(seed)

        config = Dict{String,Any}(
            "model" => Dict{String,Any}(
                "arch_type" => string(arch_type),
                "k" => _k,
                "num_embeddings" => Dict{String,Any}(
                    "type" => "PLREmbeddings",
                    "n_frequencies" => n_frequencies,
                    "frequency_scale" => frequency_scale,
                    "d_embedding" => _d_embedding,
                    "lite" => true,
                ),
                "backbone" => Dict{String,Any}(
                    "type" => "MLP",
                    "n_blocks" => _n_blocks,
                    "d_block" => _d_block,
                    "dropout" => _dropout,
                ),
            ),
            "training" => Dict{String,Any}(
                "lr" => _lr,
                "weight_decay" => _wd,
                "n_bins" => 2,
            ),
            "general" => Dict{String,Any}(),
        )

        push!(hyper_list, Dict{Symbol,Any}(
            :model_type => "tabm",
            :config => config,
            :cat_policy => "indices",
            :num_policy => "none",
            :normalization => normalization,
            :nrounds => nrounds,
            :batchsize => _batchsize,
            :seed => _seed,
            :use_float => use_float,
            :arch_config => config["model"],
            :lr => _lr,
            :wd => _wd,
        ))
    end

    rng = Xoshiro(123)
    return sample(rng, hyper_list, hyper_size, replace=false)
end

function run_experiment(
    ::Val{:TALENT},
    data,
    hyper_list;
    metrics=[:logloss, :accuracy],
    save_root=joinpath("results", "talent_checkpoints"),
)
    target_name = data[:target_name]
    feature_names = data[:feature_names]
    x_train = _matrix(data[:dtrain], feature_names)
    y_train = _target(data[:dtrain], target_name)
    x_eval = _matrix(data[:deval], feature_names)
    y_eval = _target(data[:deval], target_name)
    x_test = _matrix(data[:dtest], feature_names)
    y_test = _target(data[:dtest], target_name)

    results = OrderedDict{Symbol,Any}[]

    for (i, hyper) in enumerate(hyper_list)
        @info "run_experiment(TALENT) loop $i"
        save_path = joinpath(save_root, _talent_model_type(hyper), "hyper_$i")
        input_path = joinpath(save_path, "inputs")
        mkpath(input_path)

        x_train_path = joinpath(input_path, "x_train.csv")
        y_train_path = joinpath(input_path, "y_train.csv")
        x_eval_path = joinpath(input_path, "x_eval.csv")
        y_eval_path = joinpath(input_path, "y_eval.csv")
        x_test_path = joinpath(input_path, "x_test.csv")
        y_test_path = joinpath(input_path, "y_test.csv")
        pred_eval_path = joinpath(input_path, "pred_eval.csv")
        pred_test_path = joinpath(input_path, "pred_test.csv")
        result_path = joinpath(input_path, "result.txt")
        spec_path = joinpath(input_path, "spec.json")

        writedlm(x_train_path, x_train, ',')
        writedlm(y_train_path, y_train, ',')
        writedlm(x_eval_path, x_eval, ',')
        writedlm(y_eval_path, y_eval, ',')
        writedlm(x_test_path, x_test, ',')
        writedlm(y_test_path, y_test, ',')

        run_info = _run_talent_python(
            Dict{String,Any}(
                "spec_path" => spec_path,
                "model_type" => hyper[:model_type],
                "config" => hyper[:config],
                "x_train_path" => x_train_path,
                "y_train_path" => y_train_path,
                "x_eval_path" => x_eval_path,
                "y_eval_path" => y_eval_path,
                "x_test_path" => x_test_path,
                "y_test_path" => y_test_path,
                "pred_eval_path" => pred_eval_path,
                "pred_test_path" => pred_test_path,
                "result_path" => result_path,
                "batch_size" => hyper[:batchsize],
                "max_epoch" => hyper[:nrounds],
                "seed" => hyper[:seed],
                "save_path" => save_path,
                "cat_policy" => hyper[:cat_policy],
                "num_policy" => hyper[:num_policy],
                "normalization" => hyper[:normalization],
                "use_float" => hyper[:use_float],
            )
        )
        p_eval = vec(Float64.(readdlm(pred_eval_path, ',')))
        p_test = vec(Float64.(readdlm(pred_test_path, ',')))

        res = OrderedDict{Symbol,Any}(
            :model_type => "TALENT-" * _talent_model_type(hyper),
            :hyper_id => i,
            :train_time => parse(Float64, run_info["train_time"]),
            :best_nround => parse(Int, run_info["best_nround"]),
            :device => run_info["device"],
            :backend => "torch",
            :batchsize => hyper[:batchsize],
            :epochs => hyper[:nrounds],
            :seed => hyper[:seed],
            :arch_config => string(hyper[:arch_config]),
            :embedding_config => string(get(hyper[:arch_config], "num_embeddings", missing)),
            :lr => hyper[:lr],
            :wd => hyper[:wd],
        )

        for metric in metrics
            fun = metric_dict[metric]
            res[Symbol("eval_", metric)] = fun(p_eval, y_eval)
            res[Symbol("test_", metric)] = fun(p_test, y_test)
        end

        push!(results, res)
    end

    return DataFrame(results)
end
