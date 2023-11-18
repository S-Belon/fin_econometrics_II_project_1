"""
    HAR function
"""

function HAR_regressors(data)

    RVd = data[22:end-1] # days
    RVw = zeros(length(RVd)) # weeks
    for i in 22:(length(data)-1)
       temp = 0
       for h in 0:4
           temp = temp + data[i-h]
       end
       RVw[i-21]= temp / 5
    end
    
    RVm = zeros(length(RVd)) # months
    for i in 22:(length(data)-1) 
       temp = 0
       for h in 0:21
           temp = temp + data[i-h]
       end
       RVm[i-21] = temp / 22;
    end

    return [RVd RVw RVm]
end

"""
    OLS with constant
"""

function OLSestimatorconst(y,x)
    x = [ones(size(x)[1]) x]
    return (transpose(x) * x) \ (transpose(x) * y)
end

"""
    HAR Model
"""

function har_model(params, data)
    # Extract parameters
    beta0, beta1, beta2, beta3 = params
    
    # Extract data
    rv_d, rv_w, rv_m, rv_t = data[:, 2], data[:, 3], data[:, 4], data[:, 1]
    
    # Calculate predicted values
    y_pred = beta0 .+ beta1 .* rv_d .+ beta2 .* rv_w .+ beta3 .* rv_m
    
    # Calculate residuals
    residuals = rv_t .- y_pred
    
    # Calculate standard deviation of residuals
    std_residuals = std(residuals)
    
    # Calculate log-likelihood using the normal distribution formula
    log_likelihood = sum(-0.5 * log(2π) .- log(std_residuals) .- 0.5 * (residuals / std_residuals).^2)
    
    return -log_likelihood  # Minimize the negative log-likelihood
end

"""
    HAR prediction
"""

function predict_har(params, features)
    intercept, coeff1, coeff2, coeff3 = params
    return intercept .+ coeff1 .* features[!, 1] .+ coeff2 .* features[!, 2] .+ coeff3 .* features[!, 3]
end

"""
    train FFNN model 
"""
function trainW3(x_train, y_train, x_valid, y_valid; nodes=[5,2], eta=0.001, n_epochs=100)
    #data


    # xtrain (3, T)
    # ytrain (1, T)

    # model
    neural_net = Chain(
                        Dense(size(x_train,1),  nodes[1]), Dropout(0.2),
                        Dense(nodes[1], nodes[2]),
                        Dense(nodes[2], size(y_train,1)),
    )

    # loss
    # loss(x, y; model = neural_net) = Flux.Losses.mse(model(x), y)
    loss(x, y) = Flux.Losses.mse(neural_net(x), y)

    # optimization
    opt = Descent(eta)
    # opt = ADAM(eta)

    # params
    my_params = Flux.params(neural_net)
    orig_params = deepcopy(my_params)

    # reporting
    losses_train = []
    losses_valid = []

    # Train loop over the data
    for epoch in 1:n_epochs
        # training
        Flux.train!(loss, my_params, [(x_train, y_train)], opt)
        # reporting
        push!(losses_train, loss(x_train, y_train))
        push!(losses_valid, loss(x_valid, y_valid))
        epoch % 20 == 0 ? println("Epoch $epoch \t Loss: ", losses_train[end], " \t Test: ", losses_valid[end]) : nothing
    end

    return neural_net, losses_train, losses_valid
end

"""
    Training RNN model 
"""

function trainW4(x_train, y_train, x_valid, y_valid; nodes=[20,10], eta=0.001, n_epochs=100, verbose=20, maxpatience=20, drop=0.0, lambdaW=0.0f0)
    #data

    n_in = size(x_train,1)
    n_out = size(y_train,1)

    # model
    neural_net = Flux.Chain(LSTM(n_in,  nodes[1]), Dropout(drop),
                        Dense(nodes[1], nodes[2]), Dropout(drop),
                        Dense(nodes[2], n_out))

    # loss
    # loss(x, y; model = neural_net) = Flux.Losses.mse(model(x), y)
    function loss(x, y)
        return Flux.Losses.mse(neural_net(x), y)
    end

    # function loss_eval(x, y; model = neural_net)
    #     error1 = let
    #         Flux.reset!(model)
    #         return Flux.Losses.mse(model(x), y)
    #     end
    #     return error1
    # end

    # optimization
    # opt = Descent(eta)
    # opt = ADAM(eta)
    opt = AdamW(eta, (0.9, 0.999), lambdaW)

    # params
    my_params = Flux.params(neural_net)
    orig_params = deepcopy(my_params)

    best_loss = Inf
    best_model = deepcopy(neural_net)
    count_patience = 0

    # reporting
    losses_train = []
    losses_valid = []

    # Train loop over the data
    for epoch in 1:n_epochs

        # reset hidden state every epoch before training since we begin at t=1
        Flux.reset!(neural_net)

        # training
        Flux.train!(loss, my_params, [(x_train, y_train)], opt)

        # reporting
        # now we don't reset the network and use hidden for validation loss
        push!(losses_valid, loss(x_valid, y_valid))
        # now we reset the network and run it through train data
        Flux.reset!(neural_net)
        push!(losses_train, loss(x_train, y_train))

        # and if saving the best model, we have the one with hidden states after the train data went through
        if best_loss > losses_valid[end]
            best_model = deepcopy(neural_net)
            best_loss = losses_valid[end]
        else
            count_patience += 1
            if verbose > 1
                println("  ⊚ Counted +1 in patience, $(count_patience)/$maxpatience \t Epoch: $epoch")
            end
        end

        # reporting losses, if > 1, verbose states number of epochs to report
        (epoch % verbose == 1) || (epoch == n_epochs) ? println("Epoch [$epoch/$n_epochs] \t Training Loss $(round(losses_train[end]; digits=2)) \t Validation Loss $(round(losses_valid[end]; digits=2))") : nothing

        if count_patience >= maxpatience
            println(" □ Epoch $epoch \t Patience is LOST :) ")
            return neural_net, best_model, losses_train, losses_valid
        end
    end

    return neural_net, best_model, losses_train, losses_valid
end