# Copyright (c) 2017 Ben Zimmer. All rights reserved.

# Plot loss and related measurements for monitoring deep learning.

inputFilename <- "log.txt"


plotLoss <- function(inputFilename) {
  d <- read.csv(inputFilename, header = FALSE)
  
  colnames(d) <- c("epoch", "loss", "grad", "time")
  
  par(mfrow=c(2, 1))
  with(d, {
    plot(loss ~ epoch, type = "l", lwd = 2, col = "orange")
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


while(TRUE) {
  tryCatch({
    plotLoss(inputFilename)
  },
  error = function(e) {
    cat("error reading file\n")
  })
  Sys.sleep(15)
}