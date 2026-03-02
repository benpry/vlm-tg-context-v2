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
    map(\(l) {
      l |>
        t() |>
        as_tibble()
    }) |>
    list_rbind() |>
    unnest(cols = everything()) |>
    select(A, B, C, D, E, `F`, G, H, I, J, K, L)
  logprobs_res_out <- logprobs_res |>
    select(order(colnames(logprobs_res))) |>
    rowwise() |>
    mutate(
      across(everything(), exp),
      across(everything(), \(x) x / sum(c_across(everything())))
    )

  if ("gameId.y" %in% colnames(logprobs_cleaned)) {
    logprobs_cleaned <- logprobs_cleaned |>
      rename(runId = gameId.y)
  } else if ("workerid" %in% colnames(logprobs_cleaned)) {
    logprobs_cleaned <- logprobs_cleaned |>
      mutate(runId = as.character(workerid))
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
  logprobs <- read_csv(here(OUTPUT_LOC, file_name), show_col_types = FALSE) |>
    logprobs_to_long()

  condition_name <- file_name |>
    str_replace(".*/", "") |>
    str_replace("limited_feedback_", "") |>
    str_replace("gemma", "Gemma") |>
    str_extract("^[a-z_]+(?=_)") |>
    str_replace_all("wrong_", "other-") |>
    str_replace("no_context", "no context") |>
    str_replace("backwards", "backward")
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

get_all_logprobs <- function(model_name, no_image = FALSE, float32 = FALSE, prefix = "") {
  modstr <- if (no_image) "_no_image" else if (float32) "_float32" else ""
  yoked <- read_csv(here(OUTPUT_LOC, glue("{prefix}yoked_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long()
  shuffled <- read_csv(here(OUTPUT_LOC, glue("{prefix}shuffled_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long()
  backward <- read_csv(here(OUTPUT_LOC, glue("{prefix}backward_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long() |>
    mutate(
      orig_trialNum = 71 - orig_trialNum,
      orig_repNum = 5 - orig_repNum
    )
  ablated <- read_csv(here(OUTPUT_LOC, glue("{prefix}ablated_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long()
  other_within <- read_csv(here(OUTPUT_LOC, glue("{prefix}wrong_within_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "other-within")
  other_across <- read_csv(here(OUTPUT_LOC, glue("{prefix}wrong_across_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "other-across")
  random <- read_csv(here(OUTPUT_LOC, glue("{prefix}random_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "random")
  no_context <- read_csv(here(OUTPUT_LOC, glue("{prefix}no_context_{model_name}_logprobs{modstr}.csv"))) |>
    logprobs_to_long() |>
    mutate(condition = "no context")

  bind_rows(
    yoked,
    shuffled,
    backward,
    ablated,
    other_within,
    other_across,
    random,
    no_context
  ) |>
    mutate(
      type = glue("model_{model_name |> str_to_lower() |> str_extract('^[a-z]+')}"),
      condition = factor(condition, levels = c(
        "yoked", "shuffled", "backward", "ablated",
        "other-within", "other-across", "random", "no context"
      ))
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
