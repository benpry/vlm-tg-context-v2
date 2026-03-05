## NOTE: Not reliable. We recommend downloading directly as ZIP files from OSF.
## This code is here for reference.

library(tidyverse)
library(osfr)
library(here)

project <- osf_retrieve_node("zk8gq")

project_files <- osf_ls_files(project) |>
  filter(name != "logprobs")

R.utils::mkdirs(here("data", "logprobs"))

osf_download(
  project_files,
  path = here("data", "logprobs"),
  recurse = TRUE,
  conflicts = "skip",
  verbose = TRUE
)
