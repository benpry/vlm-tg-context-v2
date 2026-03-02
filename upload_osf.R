library(tidyverse)
library(osfr)
library(here)

project <- osf_retrieve_node("zk8gq")

osf_ls_files(project)



osf_upload(project,
    path = here("data/logprobs"),
    recurse = T, conflicts = "replace", progress = T, verbose = T)

