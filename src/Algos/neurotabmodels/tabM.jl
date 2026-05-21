function get_hyper_tabm(
    hyper_size;
    loss="mse",
    metric=loss,
    device="gpu",
    backend=:zygote,
    early_stopping_rounds=5,
    nrounds=200,
    lr=1e-3,
    wd=0.0,
    arch_type=:tabm,
    k=16,
    d_block=128,
    n_blocks=3,
    dropout=0.1,
    embedding_type="piecewise",
    d_embedding=16,
    bins=16,
    batchsize=256,
    seed=123,
)

    # tunable = [:eta, :max_depth, :subsample, :colsample_bytree, :lambda, :max_bin]
    hyper_list = Dict{Symbol,Any}[]

    for _lr in _as_iter(lr), _wd in _as_iter(wd), _k in _as_iter(k), _d_block in _as_iter(d_block),
        _n_blocks in _as_iter(n_blocks), _dropout in _as_iter(dropout), _d_embedding in _as_iter(d_embedding),
        _bins in _as_iter(bins), _batchsize in _as_iter(batchsize), _seed in _as_iter(seed)

        embedding_config = if embedding_type in (:piecewise, "piecewise")
            NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.PiecewiseLinearEmbeddings(;
                d_embedding=_d_embedding,
                bins=_bins,
            ))
        elseif embedding_type in (:linear, "linear")
            NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.LinearEmbeddings(;
                d_embedding=_d_embedding,
            ))
        elseif embedding_type in (:periodic, "periodic")
            NeuroTabModels.EmbeddingLayer(num=NeuroTabModels.PeriodicEmbeddings(;
                d_embedding=_d_embedding,
                frequencies=16,
            ))
        else
            nothing
        end

        hyper = Dict(
            :arch_name => "TabMConfig",
            :arch_config => Dict(
                :arch_type => arch_type,
                :k => _k,
                :d_block => _d_block,
                :n_blocks => _n_blocks,
                :dropout => _dropout,
            ),
            :embedding_config => embedding_config,
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
