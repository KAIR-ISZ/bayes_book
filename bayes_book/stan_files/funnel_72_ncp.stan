data {
   /* ... declarations ... */
   int D;
}

parameters {
   /* ... declarations ... */
   real mu;
   real log_tau;
   array[D-2] real eta;
}

transformed parameters {
   /* ... declarations ... statements ... */
   array[D-2] real theta;
      for (i in 1:D-2) {
      theta[i] = mu+exp(log_tau)*eta[i];
   } 
}

model {
   /* ... declarations ... statements ... */
   mu ~ normal(0,1);
   log_tau ~ normal(0,5);
   eta ~ normal(0,1); 
}