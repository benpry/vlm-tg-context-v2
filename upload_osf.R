## Upload the local logprob folders (data/logprobs/<folder>/) to the OSF
## project, then verify every local file's md5 against OSF (verify_osf.R).
##
## A file is uploaded (replacing the OSF copy) only when its md5 differs from
## the copy on OSF or it is missing there; files that already match are
## reported as "unchanged" rather than re-uploaded. This is an explicit
## md5 comparison, not osfr's conflicts = "skip", which once left stale
## (pre-banana-fix) results on OSF while appearing to succeed -- and the run
## always ends with verify_osf.R checking every file. Set DRY_RUN=1 to print
## the plan without uploading anything.
##
## Requires an OSF personal access token in the OSF_PAT environment variable
## (see osfr::osf_auth()). Every failure stops the script.

library(tidyverse)
library(osfr)
library(here)

# OSF folder listings come from the API directly (osf_helpers.R): osfr's
# osf_ls_files() can drop and duplicate files in large folders.
source(here("osf_helpers.R"))

OSF_NODE <- "zk8gq"
# Only real result files are archived: never .bak backups, checkpoints, or raw responses.
RESULT_FILE_PATTERN <- "_logprobs(_no_image)?\\.csv$"
DRY_RUN <- nzchar(Sys.getenv("DRY_RUN"))

# Retry an upload a few times: OSF's upload service occasionally answers
# 403/5xx under load. Anything that still fails afterwards is an error.
upload_with_retry <- function(remote_dir, path, max_attempts = 4) {
  for (attempt in seq_len(max_attempts)) {
    result <- tryCatch(
      osf_upload(remote_dir, path = path, conflicts = "replace", progress = FALSE, verbose = FALSE),
      error = function(e) e
    )
    if (!inherits(result, "error")) {
      return(invisible(result))
    }
    if (attempt == max_attempts) {
      stop("Upload of ", basename(path), " failed after ", attempt, " attempts: ", conditionMessage(result))
    }
    wait <- 20 * attempt
    message("  upload of ", basename(path), " failed (", conditionMessage(result), "); retrying in ", wait, "s")
    Sys.sleep(wait)
  }
}

project <- osf_retrieve_node(OSF_NODE)

logprobs_dir <- here("data/logprobs")
subdirs <- list.dirs(logprobs_dir, recursive = FALSE, full.names = TRUE)
if (length(subdirs) == 0) {
  stop("No subdirectories found in ", logprobs_dir)
}

remote_dirs <- osf_ls_files(project, n_max = Inf)

walk(subdirs, \(subdir) {
  folder_name <- basename(subdir)
  remote_dir <- remote_dirs |> filter(name == folder_name)
  if (nrow(remote_dir) == 0) {
    message("Creating remote folder ", folder_name)
    remote_dir <- osf_mkdir(project, folder_name)
  } else if (nrow(remote_dir) > 1) {
    stop("More than one remote folder named ", folder_name)
  }

  local_files <- list.files(subdir, pattern = RESULT_FILE_PATTERN, full.names = TRUE)
  if (length(local_files) == 0) {
    stop("No result files matching ", RESULT_FILE_PATTERN, " in ", subdir)
  }

  remote_files <- osf_list_folder(OSF_NODE, remote_dir$id) |> filter(kind == "file")
  plan <- tibble(path = local_files, name = basename(local_files),
                 local_md5 = unname(tools::md5sum(local_files))) |>
    left_join(remote_files |> select(name, remote_md5 = md5), by = "name") |>
    mutate(action = case_when(
      is.na(remote_md5) ~ "upload (missing on OSF)",
      remote_md5 != local_md5 ~ "upload (changed)",
      TRUE ~ "unchanged"
    ))
  to_upload <- plan |> filter(action != "unchanged")
  message(folder_name, "/: ", nrow(to_upload), " of ", nrow(plan), " files to upload, ",
          sum(plan$action == "unchanged"), " already match OSF")
  walk(seq_len(nrow(to_upload)), \(i) {
    message("  ", to_upload$action[i], ": ", to_upload$name[i])
    if (!DRY_RUN) {
      upload_with_retry(remote_dir, to_upload$path[i])
      message("  uploaded ", to_upload$name[i])
    }
  })
})

if (DRY_RUN) {
  message("Dry run: nothing uploaded.")
} else {
  message("Upload finished; verifying md5s against OSF...")
  source(here("verify_osf.R"))
}
