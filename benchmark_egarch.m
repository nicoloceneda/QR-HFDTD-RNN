
% -------------------------------------------------------------------------------
% 0. PREPARE THE DATA
% -------------------------------------------------------------------------------


% Import the train, validation and test sets

symbol = 'AAPL';

Y_train = table2array(readtable(strcat("datasets/mode sl/datasets std/", symbol, "/Y_train.csv")));
Y_valid = table2array(readtable(strcat("datasets/mode sl/datasets std/", symbol, "/Y_valid.csv")));
Y_test = table2array(readtable(strcat("datasets/mode sl/datasets std/", symbol, "/Y_test.csv")));


% -------------------------------------------------------------------------------
% 1. egarch MODEL - NORMAL
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        emodel_garch = egarch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 'Gaussian');
        estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
        cond_variance_egarch = infer(estimated_egarch, [Y_train; Y_valid]);
        valid_variance_egarch = cond_variance_egarch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        valid_sigma_egarch = sqrt(valid_variance_egarch);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qn_calculator(valid_sigma_egarch, tau);                        
        losses(p, q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p, q) < minimum_loss
            
            minimum_loss = losses(p, q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


% Calculate the volatility for the test set

emodel_garch = egarch('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 'Gaussian');
estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
cond_variance_egarch = infer(estimated_egarch, [Y_train; Y_valid; Y_test]);
test_variance_egarch = cond_variance_egarch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
test_sigma_egarch = sqrt(test_variance_egarch);


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qn_calculator(test_sigma_egarch, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-N: Test set params and loss for tau:");
disp([best_p, best_q, loss]);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qn_calculator(test_sigma_egarch, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-N: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);


% -------------------------------------------------------------------------------
% 2. egarch MODEL - STUDENT
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        emodel_garch = egarch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't');
        estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
        cond_variance_egarch = infer(estimated_egarch, [Y_train; Y_valid]);
        valid_variance_egarch = cond_variance_egarch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        nu = estimated_egarch.Distribution.DoF;
        valid_sigma_egarch = sqrt(valid_variance_egarch * (nu - 2) / nu);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qt_calculator(valid_sigma_egarch, tau, nu);                        
        losses(p, q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p, q) < minimum_loss
            
            minimum_loss = losses(p, q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


% Calculate the volatility for the test set

emodel_garch = egarch('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 't');
estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
cond_variance_egarch = infer(estimated_egarch, [Y_train; Y_valid; Y_test]);
test_variance_egarch = cond_variance_egarch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
nu = estimated_egarch.Distribution.DoF;
test_sigma_egarch = sqrt(test_variance_egarch * (nu - 2) / nu); 


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qt_calculator(test_sigma_egarch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-t: Test set params and loss for tau:");
disp([best_p, best_q, loss]);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qt_calculator(test_sigma_egarch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-t: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);


% -------------------------------------------------------------------------------
% 3. egarch MODEL - STUDENT - AR
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        for r = 1:4
        
            emodel_garch = arima('ARLags', r, 'Variance', egarch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't'));
            estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
            [residual_garch, cond_variance_egarch] = infer(estimated_egarch, [Y_train; Y_valid]);
            valid_mean_garch = Y_valid - residual_garch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
            valid_variance_egarch = cond_variance_egarch(length(Y_train) + 1: length(Y_train) + length(Y_valid));
            nu = estimated_egarch.Variance.Distribution.DoF;
            valid_sigma_egarch = sqrt(valid_variance_egarch * (nu - 2) / nu);                  
        
            tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
            Q = qtar_calculator(valid_sigma_egarch, valid_mean_garch, tau, nu);                        
            losses(p,q, r) = pinball_loss_function(Y_valid, Q, tau);           
        
            if losses(p, q, r) < minimum_loss
            
                minimum_loss = losses(p, q, r);
                best_p = p;
                best_q = q;
                best_r = r;
            
            end
        
        end
        
    end
    
end


% Calculate the volatility for the test set

emodel_garch = arima('ARLags', best_r, 'Variance', egarch('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 't'));
estimated_egarch = estimate(emodel_garch, Y_train, 'Display', 'off');
[residual_garch, cond_variance_egarch] = infer(estimated_egarch, [Y_train; Y_valid; Y_test]);
test_mean_garch = Y_test - residual_garch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
test_variance_egarch = cond_variance_egarch(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
nu = estimated_egarch.Variance.Distribution.DoF;
test_sigma_egarch = sqrt(test_variance_egarch * (nu - 2) / nu); 


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qtar_calculator(test_sigma_egarch, test_mean_garch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-t + AR: Test set params and loss for tau:");
disp([best_p, best_q, best_r, loss]);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qtar_calculator(test_sigma_egarch, test_mean_garch, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("egarch-t + AR: Test set params and loss for new tau:");
disp([best_p, best_q, best_r, loss]);


% -------------------------------------------------------------------------------
% 4. FUNCTIONS
% -------------------------------------------------------------------------------


% Define a function to calculate Q for a normal distribution

function Q = qn_calculator(sigma, tau)

    z_tau = norminv(tau);
    Q = sigma * z_tau;
    
end


% Define a function to calculate Q for a student distribution

function Q = qt_calculator(sigma, tau, nu)

    z_tau = tinv(tau, nu);
    Q = sigma * z_tau;
    
end


% Define a function to calculate Q for a student distribution where returns are AR

function Q = qtar_calculator(sigma, mean, tau, nu)

    z_tau = tinv(tau, nu);
    Q = mean + sigma * z_tau;
    
end


% Define the pinball loss Function

function L = pinball_loss_function(Y_actual, Q, tau)

    error = Y_actual - Q;                               
    error_1 = tau .* error;                             
    error_2 = (tau - 1) .* error;                      
    L = mean(max(error_1, error_2), 'all');             
    
end

