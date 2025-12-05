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


function ChainRulesCore.rrule(::typeof(project_to_simplex), y::AbstractVector)
    z = project_to_simplex(y)
    
    function project_to_simplex_pullback(ȳ)
        S = z .> 0
        
        sum_S = sum(S)
        avg = sum_S > 0 ? sum(ȳ[S]) / sum_S : zero(eltype(ȳ))
        
        ∇y = S .* (ȳ .- avg)
        
        return NoTangent(), ∇y
    end
    
    return z, project_to_simplex_pullback
end