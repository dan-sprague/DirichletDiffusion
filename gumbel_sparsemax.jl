using Random
using Distributions
using StaticArrays


const D = 4
function sample_gumbel(shape...)
    μ = rand(shape...) 
    return -log.(-log.(μ .+ 1e-12))
end
sample_gumbel(1,4)

function project_to_simplex(y::SVector{D, T}) where {D, T}
    
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
    noisy_logits = (logits .+ sample_gumbel(size(logits))) / temperature

    simplex_projection(noisy_logits)

end



x = [1.,5.,2.,10.]

simplex_projection(x)


x = gumbel_sparsemax(x)
