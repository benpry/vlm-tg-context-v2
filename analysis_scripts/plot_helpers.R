theme_set(theme_bw(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank()
  ))

strip_zero <- function(x, dp = 2) {
  sprintf(glue("%.{dp}f"), x) |>
    str_replace("0\\.", "\\.")
}

COL_SCALE <- scale_color_manual(
  values = c(
    "human_original" = "#f56942",
    "human_naive" = "#e89f46",
    "model_qwen" = "#32d97a",
    "model_gemma" = "#2cd9e6",
    "model_llama" = "#4388e8",
    "model_molmo" = "#6c70e3",
    "model_kimi" = "#b36ce3"
  ),
  labels = c(
    "human_original" = "Human (original)",
    "human_naive" = "Human (naïve)",
    "model_qwen" = "Qwen 3 VL",
    "model_gemma" = "Gemma 3",
    "model_llama" = "Llama 3.2",
    "model_molmo" = "Molmo 2",
    "model_kimi" = "Kimi VL"
  ),
  limits = c(
    "human_original",
    "human_naive",
    "model_qwen",
    "model_gemma",
    "model_llama",
    "model_molmo",
    "model_kimi"
  )
)

COL_SCALE_NOLIM <- scale_color_manual(
  values = c(
    "human_original" = "#f56942",
    "human_naive" = "#e89f46",
    "model_qwen" = "#32d97a",
    "model_gemma" = "#49e6e3",
    "model_llama" = "#43a0e8",
    "model_molmo" = "#6c70e3",
    "model_kimi" = "#b36ce3"
  ),
  labels = c(
    "human_original" = "Human (original)",
    "human_naive" = "Human (naïve)",
    "model_qwen" = "Qwen 3 VL",
    "model_gemma" = "Gemma 3",
    "model_llama" = "Llama 3.2",
    "model_molmo" = "Molmo 2",
    "model_kimi" = "Kimi VL"
  )
)

COL_SCALE_FRONTIER <- scale_color_manual(
  values = c(
    "human_original" = "#f56942",
    "human_naive" = "#e89f46",
    "model_gemini" = "#1c818c",
    "model_gpt" = "#1c478c",
    "model_claude" = "#5c1c8c"
  ),
  labels = c(
    "human_original" = "Human (original)",
    "human_naive" = "Human (naïve)",
    "model_gemini" = "Gemini 3 Flash",
    "model_gpt" = "GPT 5.2",
    "model_claude" = "Claude Sonnet 4.6"
  ),
  limits = c(
    "human_original",
    "human_naive",
    "model_gemini",
    "model_gpt",
    "model_claude"
  )
)

condition_order <- c(
  "yoked", "backward", "shuffled", "random",
  "other-within", "other-across", "ablated", "no context",
  "r1 practice", "r6 practice"
)

make_accuracy_plot <- function(df, repnum_type = "matcher",
                               ref_level = 1 / 12,
                               human_dotted = FALSE) {
  p <- df |>
    mutate(
      repnum = if (repnum_type == "original") orig_repNum + 1 else matcher_repNum + 1,
      trialnum = if (repnum_type == "original") orig_trialNum + 1 else matcher_trialNum + 1,
      condition = factor(condition, levels = condition_order),
      human = ifelse(str_detect(type, "human_"), "human", "model")
    ) |>
    ggplot(aes(x = repnum, y = accuracy, col = type)) +
    geom_hline(yintercept = ref_level, lty = "dashed") +
    stat_summary(
      aes(group = interaction(gameId, type)),
      fun = mean, geom = "line", alpha = .15,
      show.legend = TRUE
    )

  if (human_dotted) {
    p <- p +
      geom_smooth(
        aes(lty = human),
        method = "loess", formula = y ~ x, se = FALSE,
        show.legend = TRUE
      ) +
      scale_linetype_manual(
        guide = "none",
        values = c(
          "human" = "dotted",
          "model" = "solid"
        )
      )
  } else {
    p <- p +
      geom_smooth(
        method = "loess", formula = y ~ x, se = FALSE,
        show.legend = TRUE
      )
  }

  p <- p +
    stat_summary(
      fun.data = mean_cl_boot, geom = "pointrange",
      show.legend = TRUE
    ) +
    scale_x_continuous(breaks = 1:6) +
    COL_SCALE +
    labs(
      x = glue("{str_to_sentence(repnum_type)} repetition number"),
      y = "Accuracy", col = "Matcher"
    )

  if (n_distinct(df$condition) > 1) {
    p <- p +
      facet_wrap(
        ~condition,
        nrow = 2,
        labeller = as_labeller(str_to_sentence)
      )
  }

  p
}
