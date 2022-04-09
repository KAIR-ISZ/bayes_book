data {
   /* ... declarations ... */
   int D;
}

parameters {
   /* ... declarations ... */
   real mu;
   real log_tau;
   array[D-2] real theta;
}

model {
   /* ... declarations ... statements ... */
   mu ~ normal(0,1);
   log_tau ~ normal(0,5);
   for (i in 1:D-2) {
      theta[i] ~ normal(mu,exp(log_tau));
   } 
}