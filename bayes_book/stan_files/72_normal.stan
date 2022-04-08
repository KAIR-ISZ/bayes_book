
parameters {
   /* ... declarations ... */
   real q_1;
   real q_2;
}

model {
   /* ... declarations ... statements ... */
   q_1 ~ normal(1,1);
   q_2 ~ normal(-1,1);

}