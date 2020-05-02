
% -------------------------------------------------------------------------------
% 0. PREPARE THE DATA
% -------------------------------------------------------------------------------


% Import the train, validation and test sets

symbol = 'FB';
elle = 200;
symbol_elle = strcat(symbol, "_", string(elle));

Y_train = table2array(readtable(strcat("data/mode sl/datasets std/", symbol_elle, "/Y_train.csv")));
Y_valid = table2array(readtable(strcat("data/mode sl/datasets std/", symbol_elle, "/Y_valid.csv")));
Y_test = table2array(readtable(strcat("data/mode sl/datasets std/", symbol_elle, "/Y_test.csv")));


% -------------------------------------------------------------------------------
% 1. GJR MODEL - NORMAL
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        model_gjr = gjr('GARCHLags', p, 'ARCHLags', q, 'Distribution', 'Gaussian');
        estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
        cond_variance_gjr = infer(estimated_gjr, [Y_train; Y_valid]);
        valid_variance_gjr = cond_variance_gjr(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        valid_sigma_gjr = sqrt(valid_variance_gjr);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qn_calculator(valid_sigma_gjr, tau);                        
        losses(p, q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p, q) < minimum_loss
            
            minimum_loss = losses(p, q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


fileID = fopen(strcat('data/mode sl/results/', symbol, '/bench_gjr.txt'),'w');
fprintf(fileID, ['BENCHMARK - Symbol :', symbol]);
fprintf(fileID, ['\n- Sequence length: %3f', elle]);

fprintf(fileID, '\nGJR-N:');
fprintf(fileID, '\n- p: %3.1f', best_p);
fprintf(fileID, '\n- q: %3.1f', best_q);


% Calculate the volatility for the test set

model_gjr = gjr('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 'Gaussian');
estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
cond_variance_gjr = infer(estimated_gjr, [Y_train; Y_valid; Y_test]);
test_variance_gjr = cond_variance_gjr(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
test_sigma_gjr = sqrt(test_variance_gjr);


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qn_calculator(test_sigma_gjr, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-N: Test set params and loss for tau:");
disp([best_p, best_q, loss]);

fprintf(fileID,' \n* Test loss (tau): %17.15f', loss);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qn_calculator(test_sigma_gjr, tau);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-N: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);

fprintf(fileID,' \n* Test loss (new tau): %17.15f', loss);


% -------------------------------------------------------------------------------
% 2. GJR MODEL - STUDENT
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        model_gjr = gjr('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't');
        estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
        cond_variance_gjr = infer(estimated_gjr, [Y_train; Y_valid]);
        valid_variance_gjr = cond_variance_gjr(length(Y_train) + 1: length(Y_train) + length(Y_valid));
        nu = estimated_gjr.Distribution.DoF;
        valid_sigma_gjr = sqrt(valid_variance_gjr * (nu - 2) / nu);                  
        
        tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
        Q = qt_calculator(valid_sigma_gjr, tau, nu);                        
        losses(p, q) = pinball_loss_function(Y_valid, Q, tau);           
        
        if losses(p, q) < minimum_loss
            
            minimum_loss = losses(p, q);
            best_p = p;
            best_q = q;
            
        end
        
    end
    
end


fprintf(fileID, '\nGJR-t:');
fprintf(fileID, '\n- p: %3.1f', best_p);
fprintf(fileID, '\n- q: %3.1f', best_q);


% Calculate the volatility for the test set

model_gjr = gjr('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 't');
estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
cond_variance_gjr = infer(estimated_gjr, [Y_train; Y_valid; Y_test]);
test_variance_gjr = cond_variance_gjr(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
nu = estimated_gjr.Distribution.DoF;
test_sigma_gjr = sqrt(test_variance_gjr * (nu - 2) / nu); 


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qt_calculator(test_sigma_gjr, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-t: Test set params and loss for tau:");
disp([best_p, best_q, loss]);

fprintf(fileID,' \n* Test loss (tau): %17.15f', loss);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qt_calculator(test_sigma_gjr, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-t: Test set params and loss for new tau:");
disp([best_p, best_q, loss]);

fprintf(fileID,' \n* Test loss (new tau): %17.15f', loss);


% -------------------------------------------------------------------------------
% 3. GJR MODEL - STUDENT - AR
% -------------------------------------------------------------------------------


% Hyperparameter selection

losses = zeros(4, 4, 4);
minimum_loss = 1e8;

for p = 1:4
    
    for q = 1:4
        
        for r = 1:4
        
            model_gjr = arima('ARLags', r, 'Variance', gjr('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't'));
            estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
            [residual_gjr, cond_variance_gjr] = infer(estimated_gjr, [Y_train; Y_valid]);
            valid_mean_gjr = Y_valid - residual_gjr(length(Y_train) + 1: length(Y_train) + length(Y_valid));
            valid_variance_gjr = cond_variance_gjr(length(Y_train) + 1: length(Y_train) + length(Y_valid));
            nu = estimated_gjr.Variance.Distribution.DoF;
            valid_sigma_gjr = sqrt(valid_variance_gjr * (nu - 2) / nu);                  
        
            tau = [0.01, 0.05: 0.05: 0.95, 0.99];                            
            Q = qtar_calculator(valid_sigma_gjr, valid_mean_gjr, tau, nu);                        
            losses(p, q, r) = pinball_loss_function(Y_valid, Q, tau);
        
            if losses(p, q, r) < minimum_loss
            
                minimum_loss = losses(p, q, r);
                best_p = p;
                best_q = q;
                best_r = r;
            
            end
        
        end
        
    end
    
end


fprintf(fileID, '\nAR-GJR-t:');
fprintf(fileID, '\n- p: %3.1f', best_p);
fprintf(fileID, '\n- q: %3.1f', best_q);
fprintf(fileID, '\n- r: %3.1f', best_r);


% Calculate the volatility for the test set

model_gjr = arima('ARLags', best_r, 'Variance', gjr('GARCHLags', best_p, 'ARCHLags', best_q, 'Distribution', 't'));
estimated_gjr = estimate(model_gjr, Y_train, 'Display', 'off');
[residual_gjr, cond_variance_gjr] = infer(estimated_gjr, [Y_train; Y_valid; Y_test]);
test_mean_gjr = Y_test - residual_gjr(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
test_variance_gjr = cond_variance_gjr(length(Y_train) + length(Y_valid) + 1: length(Y_train) + length(Y_valid) + length(Y_test));
nu = estimated_gjr.Variance.Distribution.DoF;
test_sigma_gjr = sqrt(test_variance_gjr * (nu - 2) / nu); 


% Calculate the loss for the tau set

tau = [0.01, 0.05: 0.05: 0.95, 0.99];
Q = qtar_calculator(test_sigma_gjr, test_mean_gjr, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-t + AR: Test set params and loss for tau:");
disp([best_p, best_q, best_r, loss]);

fprintf(fileID,' \n* Test loss (tau): %17.15f', loss);


% Calculate the loss for the new tau set

tau = [0.01,0.05,0.1];
Q = qtar_calculator(test_sigma_gjr, test_mean_gjr, tau, nu);
loss = pinball_loss_function(Y_test, Q, tau);

disp("GJR-t + AR: Test set params and loss for new tau:");
disp([best_p, best_q, best_r, loss]);

fprintf(fileID,' \n* Test loss (new tau): %17.15f', loss);
fclose(fileID);


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

