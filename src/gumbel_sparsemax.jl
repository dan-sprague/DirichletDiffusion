using Random
using Distributions
using StaticArrays
using Turing,Mooncake
using FillArrays
const D = 2

function sample_gumbel(shape...)
    μ = rand(shape...) 
    return -log.(-log.(μ .+ 1e-12))
end

function project_to_simplex(y::Vector{T}) where T <: Real 
    
    μ = sort(y, rev = true)

    ρ = 1
    current_sum = zero(T) 
    sum_at_ρ = zero(T)

    for j in 1:D
        current_sum += μ[j]

        if μ[j] + (1 / j) * (1 - current_sum) > 0
            ρ = j
            sum_at_ρ = current_sum
        end
    end


    λ = (1 / ρ) * (1 - sum_at_ρ) 
    
    return max.(y .+ λ, zero(T)) 
end

function gumbel_sparsemax(logits,temperature = 1.0)
    noisy_logits = (logits .+ sample_gumbel(size(logits)...)) / temperature

    project_to_simplex(noisy_logits)

end


function ChainRulesCore.rrule(::typeof(simplex_projection), y::AbstractVector)
    z = simplex_projection(y)
    
    function simplex_projection_pullback(ȳ)
        S = z .> 0
        
        sum_S = sum(S)
        avg = sum_S > 0 ? sum(ȳ[S]) / sum_S : zero(eltype(ȳ))
        
        ∇y = S .* (ȳ .- avg)
        
        return NoTangent(), ∇y
    end
    
    return z, simplex_projection_pullback
end