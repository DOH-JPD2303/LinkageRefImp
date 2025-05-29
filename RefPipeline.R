# Load packages
library(data.table)
library(RecordLinkage)
library(e1071)
library(randomForest)
library(stringdist)

# Load data, convert to data.table
data(RLdata10000)
recs <- data.table(RLdata10000)

# Universal type conversions, missingness handling, and assigning persistent IDs
recs[, `:=` (
        # Convert to string
        by = paste0(by),
        bd = paste0(bd),
        bm = paste0(bm),

        # Record and person identifiers
        row_id = .I,
        person_id = identity.RLdata10000,

        # Replace NA with blank strings
        fname_c1 = fcoalesce(fname_c1, ''),
        fname_c2 = fcoalesce(fname_c2, ''),
        lname_c1 = fcoalesce(lname_c1, ''),
        lname_c2 = fcoalesce(lname_c2, '')
)]

# Row pairs can come in a jumbled order
# This function subsets/deduplicates the dataset so only unique pairs remain
# The resulting df will include only row ID's, where the lowest ID in each
# pair is listed first.
row_id_dedup <- function(row_id, i.row_id) {
        df <- data.table('row_id' = row_id, 'i.row_id' = i.row_id)
        df[, `:=` (
                row_id = pmin(row_id, i.row_id),
                i.row_id = pmax(row_id, i.row_id)
        )]
        df <- unique(df)
        return(df[!(row_id == i.row_id)])
}

# This is the source of truth: will come in useful in evaluation
truth <- recs[recs, on='person_id', nomatch=0][row_id != i.row_id]
truth <- row_id_dedup(truth$row_id, truth$i.row_id)
truth <- truth[recs, on='row_id', nomatch=0]
truth <- truth[recs, on=c('i.row_id' = 'row_id', 'person_id'), nomatch=0]

# Blocking ---------------------------------------------------------------------
# Exact on birth year
join_cols = c('by')
dob = recs[recs, on=join_cols, allow.cartesian=TRUE][row_id != i.row_id]
dob[, `:=` (
        fname_diff = stringdist(fname_c1, i.fname_c1, method='lv'),
        lname_diff = stringdist(lname_c1, i.lname_c1, method='lv')
)]
dob <- dob[fname_diff <= 2 | lname_diff <= 2]

# Exact on first name
join_cols = c('fname_c1', 'fname_c2')
first = recs[recs, on=join_cols, allow.cartesian=TRUE][row_id != i.row_id]
first <- first[stringdist(lname_c1, i.lname_c1, method='lv') <= 2]

# Exact on last name
join_cols = c('lname_c1', 'lname_c2')
last = recs[recs, on=join_cols, allow.cartesian=TRUE][row_id != i.row_id]
last <- last[stringdist(fname_c1, i.fname_c1, method='lv') <= 2]

# Join all candidates to a single list and deduplicate
keep_cols <- c('row_id', 'i.row_id')
candidates <- rbindlist(list(
        dob[, ..keep_cols], last[, ..keep_cols], first[, ..keep_cols]
))
candidates <- row_id_dedup(candidates$row_id, candidates$i.row_id)

# Add the other columns for feature engineering
candidates <- candidates[recs, on='row_id', nomatch=0]
candidates <- candidates[recs, on=c('i.row_id' = 'row_id'), nomatch=0]

# We have labeled data- use it to apply a yes/no indicator.
# This is our model target
candidates[, is_match := as.factor(person_id == i.person_id)]

print_blocking_performance <- function(df, candidates, truth) {
        # Identify cases where truth and candidates agree
        agree <- candidates[truth, on = c('row_id', 'i.row_id'), nomatch=0]
        pct_agree <- formatC(100 * nrow(agree) / nrow(truth), format="f", digits = 2)
        msg <- paste0(
                "Number of true matches found: ",
                nrow(agree),
                " (", pct_agree, "%)\n"
        )
        cat(msg)

        # How much have we reduced the search space?
        num_possible_pairs <- choose(nrow(df), 2)
        reduction_ratio <- 100 * (
                num_possible_pairs - nrow(candidates)
        ) / num_possible_pairs
        reduction_ratio <- formatC(reduction_ratio, format="f", digits = 2)
        msg <- paste0("Reduction ratio: (", reduction_ratio, "%)\n")
        cat(msg)
}

print_blocking_performance(recs, candidates, truth)

# Feature engineering ----------------------------------------------------------
column_pairs <- list(
        # LHS first name, first part
        c("fname_c1", "i.fname_c1"),
        c("fname_c1", "i.fname_c2"),
        c("fname_c1", "i.lname_c1"),
        c("fname_c1", "i.lname_c2"),

        # LHS first name, second part
        c("fname_c2", "i.fname_c1"),
        c("fname_c2", "i.fname_c2"),
        c("fname_c2", "i.lname_c1"),
        c("fname_c2", "i.lname_c2"),

        # LHS last name, first part
        c("lname_c1", "i.fname_c1"),
        c("lname_c1", "i.fname_c2"),
        c("lname_c1", "i.lname_c1"),
        c("lname_c1", "i.lname_c2"),

        # LHS last name, second part
        c("lname_c2", "i.fname_c1"),
        c("lname_c2", "i.fname_c2"),
        c("lname_c2", "i.lname_c1"),
        c("lname_c2", "i.lname_c2"),

        # Birth date (3 parts)
        c("by", "i.by"),
        c("bm", "i.bm"),
        c("bd", "i.bd")
)

# Apply Jaro-Winkler similarity to each pair
jw_names <- paste0(
        "JW_", sapply(column_pairs, function(x) paste0(x[1], "_", x[2]))
)
candidates[, (jw_names) := lapply(
        column_pairs,
        function(cols) stringdist::stringdist(get(cols[1]), get(cols[2]), method='jw')
)]

# Train test split -------------------------------------------------------------
# Set seed for reproducibility
set.seed(123)

# Controls percentage of records in training set vs. testing
train_ratio <- 0.80

# Create training and testing indices
train_indices <- sample(
        seq_len(nrow(candidates)), size = train_ratio * nrow(candidates)
)
train <- candidates[train_indices]
test <- candidates[-train_indices]

# Extract model features and outputs from each set
X_train <- train[, ..jw_names]
Y_train <- train$is_match
X_test <- test[, ..jw_names]
Y_test <- test$is_match

# Model fit --------------------------------------------------------------------
# SVM model - train if needed, otherwise load pre-existing
svm_mod_fn <- file.path('./models/svm_mod.RDS')
if(!dir.exists(dirname(svm_mod_fn)) {
        dir.create(dirname(svm_mod_fn))
}
if(!file.exists(svm_mod_fn)) {
        svm_mod <- svm(y = Y_train, x = X_train, probability=TRUE)
        saveRDS(svm_mod, svm_mod_fn)
} else {
        svm_mod <- readRDS(svm_mod_fn)
}

# RF model - train if needed, load if not
rf_mod_fn <- file.path('./models/rf_mod.RDS')
if(!file.exists(rf_mod_fn)) {
        rf_mod <- randomForest(y = Y_train, x = X_train)
        saveRDS(rf_mod, rf_mod_fn)
} else {
        rf_mod <- readRDS(rf_mod_fn)
}

# Meta model - train or load
meta_mod_fn <- file.path('./models/meta_model.RDS')
if(!file.exists(meta_mod_fn)) {
        # Get training set predictions to fit the meta model
        svm_train_preds <- predict(svm_mod, X_train, probability=TRUE)
        svm_train_probs <- attr(svm_train_preds, "probabilities")
        rf_train_preds <- predict(rf_mod, X_train, type="prob")

        # Input data for meta model
        metadata <- data.table(
                svm_t = svm_train_probs[, 2],
                rf_t = rf_train_preds[, 2],
                is_match = Y_train
        )

        # Fit model and save
        meta_model <- glm(is_match ~ ., data = metadata, family = "binomial")
        saveRDS(meta_model, meta_mod_fn)
} else {
        meta_model <- readRDS(meta_mod_fn)
}

# Model eval -------------------------------------------------------------------
# Custom function to run the whole inference pipeline
inference <- function(X, svm_mod, rf_mod, meta_mod) {
        svm_train_preds <- predict(svm_mod, X, probability=TRUE)
        svm_train_probs <- attr(svm_train_preds, "probabilities")
        rf_train_preds <- predict(rf_mod, X, type="prob")
        metadata <- data.table(
                svm_t = svm_train_probs[, 2],
                rf_t = rf_train_preds[, 2]
        )
        return(predict(meta_model, metadata, type="response"))
}

# Gets confusion matrix and evaluation stats
compute_metrics <- function(predicted, actual, positive_class) {
        # Convert predictions and actual labels to factors for consistency
        predicted <- factor(predicted, levels = levels(actual))
        actual <- factor(actual, levels = levels(actual))

        # Confusion matrix
        confmat <- table(Predicted = predicted, Actual = actual)

        # Extract true/false positives and negatives
        TP <- confmat[positive_class, positive_class]
        FP <- sum(confmat[positive_class, ]) - TP
        FN <- sum(confmat[, positive_class]) - TP
        TN <- sum(confmat) - TP - FP - FN

        # Output metrics
        PPV <- TP / (TP + FP)
        NPV <- TN / (TN + FN)
        Sensitivity <- TP / (TP + FN) 
        Specificity <- TN / (TN + FP)
        F1 <- 2 * TP / (2 * TP + FP + FN)

        # Return results as a list
        return(list(
                Confusion = confmat,
                PPV = PPV,
                Sensitivity = Sensitivity,
                Specificity = Specificity,
                F1 = F1
        ))
}

# Get predictions on test set and evaluate
meta_preds <- inference(X_test, svm_mod, rf_mod, meta_model)
meta_metrics <- compute_metrics(as.factor(meta_preds > 0.5), Y_test, "TRUE")
print(meta_metrics)

# Save candidates to file for the performance snippets script
candidates_fn <- file.path("./data/candidates.RDS")
if(!dir.exists(dirname(candidates_fn)) {
        dir.create(dirname(candidates_fn))
}
if (!file.exists(candidates_fn)) {
        saveRDS(candidates, candidates_fn)
}

