logprobs_to_long <- function(logprobs) {
  if (!"orig_trialNum" %in% colnames(logprobs)) {
    logprobs <- logprobs |>
      mutate(
        orig_trialNum = trialNum,
        orig_repNum = repNum,
        matcher_trialNum = trialNum,
        matcher_repNum = repNum
      )
  }

  logprobs_cleaned <- logprobs |>
    mutate(
      model_logprobs = map(model_logprobs, \(l) {
        if (is.na(l)) {
          return(NA)
        }
        l |>
          str_replace_all("'", '"') |>
          fromJSON()
      }),
      message = map(message, fromJSON),
      message_history = message_history |>
        replace_na("[]") |>
        map(fromJSON)
    )

  logprobs_msglens <- logprobs_cleaned$message |>
    map_int(\(m) {
      m$text |>
        replace_na("") |>
        str_c(collapse = " ") |>
        str_count("\\S+")
    })
  logprobs_ctxlens <- logprobs_cleaned$message_history |>
    map_int(\(m) {
      if (length(m) == 0) {
        return(0L)
      }

      m |>
        list_rbind() |>
        pull(text) |>
        replace_na("") |>
        str_c(collapse = " ") |>
        str_count("\\S+")
    })

  logprobs_res <- logprobs_cleaned$model_logprobs |>
    map(as_tibble_row) |>
    list_rbind() |>
    select(A, B, C, D, E, `F`, G, H, I, J, K, L)
  logprobs_res_out <- logprobs_res |>
    rowwise() |>
    mutate(
      across(everything(), exp),
      across(everything(), \(x) x / sum(c_across(everything()), na.rm = TRUE)),
      across(everything(), \(x) replace_na(x, 0))
    )

  if ("gameId.y" %in% colnames(logprobs_cleaned)) {
    logprobs_cleaned <- logprobs_cleaned |>
      rename(runId = gameId.y)
  } else if ("workerid" %in% colnames(logprobs_cleaned)) {
    logprobs_cleaned <- logprobs_cleaned |>
      mutate(runId = as.character(workerid))
  } else if ("shuffle_rep" %in% colnames(logprobs_cleaned)) {
    logprobs_cleaned <- logprobs_cleaned |>
      mutate(runId = as.character(shuffle_rep))
  } else {
    logprobs_cleaned <- logprobs_cleaned |>
      mutate(runId = "1")
  }

  logprobs_combined <- logprobs_cleaned |>
    select(
      gameId, runId, orig_trialNum, orig_repNum,
      matcher_trialNum, matcher_repNum, target, condition
    ) |>
    cbind(logprobs_res_out) |>
    pivot_longer(
      cols = c(A:L),
      names_to = "tangram",
      values_to = "logprob"
    ) |>
    # filter(target == tangram) |>
    rename(prob = logprob) |>
    select(
      gameId, runId, condition, orig_trialNum, orig_repNum,
      matcher_trialNum, matcher_repNum, target,
      selection = tangram, prob
    ) |>
    mutate(
      message_length = rep(logprobs_msglens, each = 12),
      context_length = rep(logprobs_ctxlens, each = 12)
    )

  logprobs_combined
}

load_logprobs <- function(file_name) {
  logprobs <- read_csv(here("data", "logprobs", file_name), show_col_types = FALSE) |>
    logprobs_to_long()

  condition_name <- file_name |>
    str_replace(".*/", "") |>
    str_replace("limited_feedback_", "") |>
    str_replace("gemma", "Gemma") |>
    str_extract("^[a-z_]+[16]?(?=_)") |>
    str_replace_all("wrong_", "other-") |>
    str_replace("no_context", "no context") |>
    str_replace("backwards", "backward")
  if (condition_name %in% c("r1", "r6")) {
    condition_name <- str_c(condition_name, " practice")
  }
  if (condition_name == "backward") {
    logprobs <- logprobs |>
      mutate(
        orig_trialNum = 71 - orig_trialNum,
        orig_repNum = 5 - orig_repNum
      )
  }

  model_name <- file_name |>
    str_extract("(?<=_)([A-Za-z0-9.-]+)(?=_logprobs)")
  feedback_type <- file_name |>
    str_extract("^.*(?=/)") |>
    str_replace_all(c(
      "full_feedback" = "full",
      "human_yoked" = "limited human-yoked",
      "interactive" = "limited interactive"
    ))
  image_type <- if (str_detect(file_name, "no_image")) {
    "no image"
  } else {
    "image"
  }

  logprobs |>
    mutate(
      file_name = file_name,
      model_name = model_name,
      type = glue("model_{model_name |> str_to_lower() |> str_extract('^[a-z]+')}"),
      condition = condition_name,
      feedback = feedback_type,
      image = image_type
    )
}

filter_logprobs <- function(logprobs) {
  logprobs |>
    filter(target == selection) |>
    rename(accuracy = prob) |>
    select(-selection)
}

cast_random <- function(logprobs) {
  logprobs |>
    arrange(type, condition, runId, matcher_trialNum) |>
    group_by(type, condition, runId) |>
    mutate(gameId = ifelse(condition != "random", gameId,
      gameId[1]
    )) |> # arbitrary casting for plotting
    ungroup()
}

calculate_accuracies <- function(logprobs, join_only = FALSE) {
  acc <- logprobs
  prev_acc <- logprobs
  if (!join_only) {
    acc <- logprobs |>
      filter_logprobs() |>
      cast_random() |>
      arrange(type, condition, gameId, runId, matcher_trialNum) |>
      group_by(type, condition, gameId, runId, target) |>
      mutate(target_repNum = ifelse(condition %in% c("shuffled", "random"), 0:5, matcher_repNum))
    prev_acc <- logprobs |>
      group_by(
        type, condition, gameId, runId, orig_trialNum, orig_repNum,
        matcher_trialNum, matcher_repNum, target
      ) |>
      filter(prob == max(prob, na.rm = TRUE)) |>
      summarise(
        accuracy = max(selection == target, na.rm = TRUE),
        .groups = "drop"
      ) |>
      cast_random() |>
      arrange(type, condition, gameId, runId, matcher_trialNum) |>
      group_by(type, condition, gameId, runId, target) |>
      mutate(target_repNum = ifelse(condition %in% c("shuffled", "random"), 0:5, matcher_repNum))
  }

  acc_next <- acc |>
    left_join(
      prev_acc |> mutate(target_repNum = target_repNum + 1) |>
        select(type, gameId, runId, condition,
          target_repNum, target,
          prev_accuracy = accuracy
        ),
      by = join_by(type, gameId, runId, condition, target_repNum, target)
    )

  acc_next
}
