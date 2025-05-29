# Load packages
library(data.table)
library(RecordLinkage)
library(stringdist)
library(microbenchmark)
library(snow)
library(foreach)
library(doParallel)
library(parallel)

candidates <- readRDS('./data/candidates.RDS')
candidates[, `:=` (
        DOB = as.Date(paste(by, bm, bd, sep="-"), format="%Y-%m-%d"),
        i.DOB = as.Date(paste(i.by, i.bm, i.bd, sep="-"), format="%Y-%m-%d")
)]
big_candidates <- rbindlist(rep(list(candidates), 10))

# Avoid Rowwise Operations for Vectorized Functions ----------------------------
timing <- microbenchmark(
        bygroup = candidates[, 'DOB_HAM' := stringdist(DOB, i.DOB, method="hamming"), by=.I],
        noby = candidates[, 'DOB_HAM2' := stringdist(DOB, i.DOB, method="hamming")],
        times = 3
        )
print(timing)
print(identical(candidates$DOB_HAM, candidates$DOB_HAM2))

# Make a sufficiently large dataset for demonstration
timing <- microbenchmark(
        bygroup = big_candidates[, 'DOB_HAM' := stringdist(DOB, i.DOB, method="hamming"), by=.I],
        noby = big_candidates[, 'DOB_HAM2' := stringdist(DOB, i.DOB, method="hamming")],
        times = 3
        )
print(timing)

# Use the right function for the job -------------------------------------------
# Max DOB
print(microbenchmark(
        rowwise = candidates[, MAX_DOB := max(DOB, i.DOB), by=.I],
        vectorized = candidates[, MAX_DOB2 := pmax(DOB, i.DOB)],
        times = 5
        ))
print(identical(candidates$MAX_DOB, candidates$MAX_DOB2))

# Use functions as intended ----------------------------------------------------
candidates[, 'FIRSTNAME_COS_C2' := stringdist(
        fname_c1,
        i.fname_c1,
        method = c('cosine'),
        q = ifelse(
                nchar(fname_c2) < 3 | nchar(i.fname_c2) < 3, 
                min(nchar(fname_c2), nchar(i.fname_c2)),
                3
                )
), by=.I]

# Cosine edit distance with flexible handling of small strings
cosine_qflex <- function(left, right, max_q=3) {
        # Get minimum length between each pair
        min_length <- pmin(nchar(left), nchar(right))

        # Start output vector
        out <- vector("numeric", length = length(left))

        # Populate each case
        q_seq <- seq(max_q)
        for(q in seq_along(q_seq)) {
                # For all integers up to the last, we only check for equality.
                # In the last iteration, we include everything else
                if (q == length(q_seq)) {
                        idx <- which(min_length >= q)
                } else {
                        idx <- which(min_length == q)
                }

                # Run the string distance function
                out[idx] <- stringdist::stringdist(
                        left[idx], right[idx], method = 'cosine', q = q
                )
            }
        return(out)
}

candidates[, FIRSTNAME_COS_C2_ALT := cosine_qflex(fname_c2, i.fname_c2)]

# In this case, results are different. Preview the first few instances.
examine_cols <- c('fname_c2', 'i.fname_c2', 'FIRSTNAME_COS_C2', 'FIRSTNAME_COS_C2_ALT')
print(head(
        candidates[FIRSTNAME_COS_C2_ALT != FIRSTNAME_COS_C2, ..examine_cols]
))

# Avoid foreach -------------------------------------------------------------
# Set up parallel processes
no_cores <- 6
cl <- makeCluster(no_cores, type="PSOCK")
registerDoParallel(cl)

# Split data into one group for each process
big_candidates[, grp := cut(seq_len(nrow(big_candidates)), breaks=no_cores, labels=FALSE)]
cand_split <- split(big_candidates, by='grp')

# Get time for each type
foreach_time <- system.time({
        # For-each, works best forking, which is not available on Windows
         foreach_result <- foreach(
                i = 1:no_cores,
                .packages = c('data.table', 'stringdist'),
                .export="cand_split",
                .combine=c
        ) %dopar% {
                x <- cand_split[[i]]
                output <- stringdist::stringdist(x$by, x$i.by, method='jw', nthread=1)
                return(output)
        }
})
print(foreach_time)

parlapply_time <- system.time({
        # Performs better on Windows
        parlap_result <- do.call(c, parLapply(
                cl, cand_split, 
                function(x) stringdist::stringdist(x$by, x$i.by, method='jw', nthread=1)
        ))
})
print(parlapply_time)

print(table(parlap_result == foreach_result))

stopCluster(cl)
stopImplicitCluster()
rm(cl)

