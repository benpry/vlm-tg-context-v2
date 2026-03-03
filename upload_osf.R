library(tidyverse)
library(osfr)
library(here)

project <- osf_retrieve_node("zk8gq")

osf_ls_files(project)



logprobs_dir <- here("data/logprobs")
subdirs <- list.dirs(logprobs_dir, recursive = FALSE, full.names = TRUE)
conflicts_mode <- Sys.getenv("OSF_CONFLICTS", "replace")
allowed_modes <- c("skip", "replace")

if (!conflicts_mode %in% allowed_modes) {
  stop("OSF_CONFLICTS must be one of: ", paste(allowed_modes, collapse = ", "))
}

walk(subdirs, \(subdir) {
  uploader <- \(mode) {
    osf_upload(
      project,
      path = subdir,
      recurse = TRUE,
      conflicts = mode,
      progress = TRUE,
      verbose = TRUE
    )
  }

  if (identical(conflicts_mode, "skip")) {
    uploader("skip")
    return(invisible(NULL))
  }

  tryCatch(
    uploader("replace"),
    error = \(e) {
      if (grepl("get_parent_id\\(\\)|nrow\\(x\\) == 1", conditionMessage(e))) {
        message("Replace failed for ", basename(subdir), "; retrying with conflicts='skip'.")
        uploader("skip")
      } else {
        stop(e)
      }
    }
  )
})

