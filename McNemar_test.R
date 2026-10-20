##############################
# Written in a function form #
##############################

McNemar_elo <- function(dat1, dat2){
  dat1 = dat1[,c(1:3,6:7)]
  dat2 = dat2[,6:7]
  colnames(dat2) = c('winner_old_D', 'loser_old_D')
  dat = cbind(dat1, dat2)
  n11 = n12 = n21 = n22 = 0
  for(i in 1:nrow(dat)){
    if(dat[i,4] > dat[i,5] & dat[i,6] > dat[i,7]){
      n11 = n11 + 1 
    }else if(dat[i,4] > dat[i,5] & dat[i,6] <= dat[i,7]){
      n12 = n12 + 1    # "elo" correct, "eloD" incorrect
    }else if(dat[i,4] <= dat[i,5] & dat[i,6] > dat[i,7]){
      n21 = n21 + 1    # "elo" incorrect, "eloD" correct
    }else if(dat[i,4] <= dat[i,5] & dat[i,6] <= dat[i,7]){
      n22 = n22 + 1
    }
  }
  output_dat <- matrix(c(n11, n21, n12, n22), nrow = 2,
                       dimnames = list("elo" = c("Correct", "InCorrect"),
                                       "eloD" = c("Correct", "InCorrect")))
  z0 = (n21-n12)/sqrt(n21+n12)
  pvalue = 1-pnorm(z0)
  output = list(output_dat = output_dat,
                z0 = z0,
                pvalue = pvalue)
  
  return(output)
}


#################################
# McNemar for glicko vs glickoD #
#################################

McNemar_g <- function(dat1, dat2){
  dat1 = dat1[,1:3]
  dat2 = dat2[,1:3]
  colnames(dat2) = c('winner_old_D', 'loser_old_D')
  dat = cbind(dat1, dat2)
  n11 = n12 = n21 = n22 = 0
  for(i in 1:nrow(dat)){
    if(dat[i,2] > dat[i,3] & dat[i,5] > dat[i,6]){
      n11 = n11 + 1 
    }else if(dat[i,2] > dat[i,3] & dat[i,5] <= dat[i,6]){
      n12 = n12 + 1    # "glicko" correct, "glickoD" incorrect
    }else if(dat[i,2] <= dat[i,3] & dat[i,5] > dat[i,6]){
      n21 = n21 + 1    # "glicko" incorrect, "glickoD" correct
    }else if(dat[i,2] <= dat[i,3] & dat[i,5] <= dat[i,6]){
      n22 = n22 + 1
    }
  }
  output_dat <- matrix(c(n11, n21, n12, n22), nrow = 2,
                       dimnames = list("glicko" = c("Correct", "InCorrect"),
                                       "glickoD" = c("Correct", "InCorrect")))
  z0 = (n21-n12)/sqrt(n21+n12)
  pvalue = 1-pnorm(z0)
  output = list(output_dat = output_dat,
                z0 = z0,
                pvalue = pvalue)
  
  return(output)
}

setwd("~/Documents/Latex/Hua1/code/")
