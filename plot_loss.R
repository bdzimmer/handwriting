# Copyright (c) 2017 Ben Zimmer. All rights reserved.

# Plot loss and related measurements for monitoring deep learning.

lossInputFilename <- "log_charpos.txt"
callbackInputFilename <- "log_charpos_callback.txt"


plotLoss <- function(d) {
  with(d, {
    plot(loss ~ epoch, type = "l", lwd = 2, col = "red")
    title(main = paste(
      "Loss",
      round(tail(loss, 1), 6)))
  })
  
  with(d, {
    plot(grad ~ epoch, type = "l", lwd = 2, col = "blue")
    title(main = paste(
      "Gradient",
      round(tail(grad, 1), 6)))
  })
}


plotGeneric <- function(d) {
  # assumes first column of d is epoch
  ncols <- ncol(d) - 1
  for(idx in 2:(ncols + 1)) {
    curData <- d[[idx]]
    curDataName <- colnames(d)[idx]
    plot(
      curData ~ d$epoch, type = "l", lwd = 2, col = "green",
      xlab = "epoch", ylab = curDataName)
    title(main = paste(
      curDataName,
      round(tail(curData, 1), 6)))
  }
}


while(TRUE) {
  tryCatch({
    
    par(mfcol=c(3, 2))
    
    d <- read.csv(lossInputFilename, header = FALSE)
    colnames(d) <- c("epoch", "loss", "grad", "time")
    plotLoss(d)
    plot(0, type="n")
    
    d <- read.csv(callbackInputFilename, header = FALSE)
    colnames(d) <- c("epoch", "accuracy", "roc_auc", "distance")
    plotGeneric(d)
    
  },
  error = function(e) {
    cat("error reading files or plotting\n")
  })
  Sys.sleep(15)
}