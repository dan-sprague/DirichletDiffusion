using Turing
using ChainRulesCore     
using LinearAlgebra
using Distances
using Distributions
using Random
using FillArrays

include("src/gumbel_sparsemax.jl")


function calc_logits_matrix(μ, x)
    return -0.5 .* pairwise(SqEuclidean(), μ, x, dims=2)
end

function gumbel_sparsemax_matrix(logits, noise, temperature)
    noisy_logits = (logits .+ noise) ./ temperature
    
    return mapslices(project_to_simplex, noisy_logits, dims=1)
end


function deterministic_sparsemax(logits, temperature=1.0)
    return mapslices(project_to_simplex, logits ./ temperature, dims=1)
end

@model function gumbel_sparsemax_tmm(x)
    D_feature, N = size(x)
    K = 3

    μ ~ filldist(MvNormal(Zeros(D_feature), I), K)

    logits = calc_logits_matrix(μ, x)
    
    W = deterministic_sparsemax(logits, 1.0)
    
    μ_effective = μ * W
    residuals = x .- μ_effective
    Turing.@addlogprob! sum(-0.5 .* sum(abs2, residuals, dims=1))
    
    W_stochastic = Zygote.ignore() do
        noise = sample_gumbel(K, N) 
        gumbel_sparsemax_matrix(logits, noise, 1.0)
    end


    return (W= W_stochastic,)
end


D_feature = 2
K_true = 3   

μ1 = [2.0, 2.0]
μ2 = [-2.0, -2.0]
μ3 = [2.0, -2.0]

Σ = [1.0 0.0; 0.0 1.0] 
components = [
    MvNormal(μ1, Σ),
    MvNormal(μ2, Σ),
    MvNormal(μ3, Σ)
]

priors = [0.5, 0.25, 0.25]

model = MixtureModel(components, priors)

N = 1000
data = rand(model, N)


model = gumbel_sparsemax_gmm(data)

println("Sampling...")
chain = sample(model, NUTS(500,0.65), 1000)

describe(chain) 


W = generated_quantities(model,chain)


all_Ws = [g.W for g in W] 

W_mean = mean(all_Ws)

