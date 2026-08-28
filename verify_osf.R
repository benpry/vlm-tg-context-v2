## Verify that every local result file in data/logprobs/<folder>/ matches its
## copy on OSF (same md5), and report each file's OSF version number and
## modification date. Stops with an error if any file is missing or differs.
##
## Use this after uploading, and before submitting, to make sure the archive
## linked in the paper is the data the analysis used.

library(tidyverse)
library(here)

# Folder listings come from the OSF API directly (osf_helpers.R): osfr's
# osf_ls_files() can drop and duplicate files in large folders.
source(here("osf_helpers.R"))

OSF_NODE <- "zk8gq"
RESULT_FILE_PATTERN <- "_logprobs(_no_image)?\\.csv$"

remote_dirs <- osf_list_folder(OSF_NODE) |> filter(kind == "folder")
subdirs <- list.dirs(here("data/logprobs"), recursive = FALSE, full.names = TRUE)
if (length(subdirs) == 0) {
  stop("No subdirectories found in data/logprobs")
}

remote_file_info <- function(remote_dir) {
  osf_list_folder(OSF_NODE, remote_dir$id) |>
    filter(kind == "file") |>
    select(name, remote_md5 = md5, remote_version = version, remote_modified = date_modified)
}

comparison <- map(subdirs, \(subdir) {
  folder_name <- basename(subdir)
  remote_dir <- remote_dirs |> filter(name == folder_name)
  if (nrow(remote_dir) != 1) {
    stop("Expected exactly one remote folder named ", folder_name, ", found ", nrow(remote_dir))
  }
  local_files <- list.files(subdir, pattern = RESULT_FILE_PATTERN, full.names = TRUE)
  tibble(
    folder = folder_name,
    name = basename(local_files),
    local_md5 = unname(tools::md5sum(local_files))
  ) |>
    left_join(remote_file_info(remote_dir), by = "name") |>
    mutate(status = case_when(
      is.na(remote_md5) ~ "missing on OSF",
      remote_md5 == local_md5 ~ "match",
      TRUE ~ "DIFFERENT"
    ))
}) |>
  list_rbind()

print(comparison |> count(folder, status), n = Inf)

problems <- comparison |> filter(status != "match")
if (nrow(problems) > 0) {
  print(problems |> select(folder, name, status, remote_version, remote_modified), n = Inf)
  stop(nrow(problems), " local file(s) do not match OSF; see above.")
}
message("All ", nrow(comparison), " local result files match their OSF copies.")
