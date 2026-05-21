function get_hyper_modernnca(
    hyper_size;
    loss="mse",
    metric=loss,
    device="gpu",
    backend=:zygote,
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    d_embedding=128,
    d_block=256,
    n_blocks=2,
    dropout=0.1,
    temperature=1.0,
    sample_rate=0.01,
    batchsize=256,
    seed=123,
)

    hyper_list = Dict{Symbol,Any}[]

    for _lr in _as_iter(lr), _wd in _as_iter(wd), _d_embedding in _as_iter(d_embedding),
        _d_block in _as_iter(d_block), _n_blocks in _as_iter(n_blocks), _dropout in _as_iter(dropout),
        _temperature in _as_iter(temperature), _sample_rate in _as_iter(sample_rate),
        _batchsize in _as_iter(batchsize), _seed in _as_iter(seed)

        hyper = Dict(
            :arch_name => "ModernNCAConfig",
            :arch_config => Dict(
                :d_embedding => _d_embedding,
                :d_block => _d_block,
                :n_blocks => _n_blocks,
                :dropout => _dropout,
                :temperature => _temperature,
                :sample_rate => _sample_rate,
            ),
            :loss => loss,
            :metric => metric,
            :device => device,
            :backend => backend,
            :early_stopping_rounds => early_stopping_rounds,
            :nrounds => nrounds,
            :lr => _lr,
            :wd => _wd,
            :batchsize => _batchsize,
            :seed => _seed,
        )

        push!(hyper_list, hyper)
    end
    rng = Xoshiro(123)
    hyper_list = sample(rng, hyper_list, hyper_size, replace=false)
    return hyper_list
end
