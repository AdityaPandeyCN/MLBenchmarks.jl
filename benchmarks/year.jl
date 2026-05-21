using MLBenchmarks
import MLBenchmarks: mse, rmse, mae, logloss, accuracy, gini, ndcg
import MLBenchmarks: run_experiment

using DataFrames
using CSV
using Statistics: mean, std
using StatsBase: sample
using OrderedCollections
using Zygote

data_name = :year
tabm_hyper_size = 16
modernnca_hyper_size = 8
data = load_data(data_name; uniformize=true)
metrics = [:mse, :rmse]
mkpath(joinpath("results", string(data_name)))

################################
# TabM
################################
hyper_list = MLBenchmarks.get_hyper_tabm(tabm_hyper_size; data.loss, data.metric, device=:gpu, backend=:zygote, nrounds=200, early_stopping_rounds=2, lr=1e-3, arch_type=:tabm, k=[8, 16, 32], d_block=[32, 64, 128], n_blocks=2:3, dropout=0.1, batchsize=1024, seed=123, embedding_type=:piecewise, d_embedding=[8, 16], bins=[16, 32])
results_df = run_experiment(:NeuroTabModels, data, hyper_list; metrics)
CSV.write(joinpath("results", string(data_name), "tabm.csv"), results_df)

hyper_list = MLBenchmarks.get_hyper_talent_tabm(tabm_hyper_size; nrounds=200, lr=1e-3, arch_type=:tabm, k=[8, 16, 32], d_block=[32, 64, 128], n_blocks=2:3, dropout=0.1, batchsize=1024, seed=123, d_embedding=[8, 16], normalization="none")
results_df = run_experiment(:TALENT, data, hyper_list; metrics, save_root=joinpath("results", string(data_name), "talent_checkpoints"))
CSV.write(joinpath("results", string(data_name), "talent_tabm.csv"), results_df)

################################
# ModernNCA
################################
hyper_list = MLBenchmarks.get_hyper_modernnca(modernnca_hyper_size; data.loss, data.metric, device=:gpu, backend=:zygote, nrounds=200, early_stopping_rounds=2, lr=1e-3, d_embedding=[64, 128], d_block=[128, 256], n_blocks=[1, 2], dropout=0.1, temperature=1.0, sample_rate=0.01, batchsize=1024, seed=123)
results_df = run_experiment(:NeuroTabModels, data, hyper_list; metrics)
CSV.write(joinpath("results", string(data_name), "modernnca.csv"), results_df)

hyper_list = MLBenchmarks.get_hyper_talent_modernnca(modernnca_hyper_size; nrounds=200, lr=1e-3, dim=[64, 128], d_block=[128, 256], n_blocks=[1, 2], dropout=0.1, temperature=1.0, sample_rate=0.01, batchsize=1024, seed=123, normalization="none")
results_df = run_experiment(:TALENT, data, hyper_list; metrics, save_root=joinpath("results", string(data_name), "talent_checkpoints"))
CSV.write(joinpath("results", string(data_name), "talent_modernnca.csv"), results_df)
