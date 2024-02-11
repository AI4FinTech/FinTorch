**Dataset Addition Merge Checklist**

**Please ensure the following are completed before merging this pull request:**

- [ ] Contributor signed the license agreement: To contribute to our repository, the contributor must sign the license agreement
- [ ] Thorough Description: The dataset is adequately described in the issue, including origin, content, size, licensing, and the relevant academic paper.
- [ ] Script Functionality:
  - [ ] Download and extraction script downloads the dataset to the intended location.
  - [ ] The script correctly preprocesses the data into a PyTorch Lightning-compatible format.
- [ ] Tests: Tests pass and successfully verify:
  - [ ] Correct download and extraction.
  - [ ] Dataset loading in PyTorch Lightning.
  - [ ] Sample data integrity (e.g., shape, types).
- [ ] Tutorial:
  - [ ] Clear notebook guide on how to use the download/extraction script.
  - [ ] Instructions on loading the dataset in PyTorch Lightning.
  - [ ] A simple training/inference example.
- [ ] Considerations:
  - [ ] Licensing is compatible with the repository.
  - [ ] If applicable, a strategy for maintaining up-to-date compatibility with the dataset is discussed.
  - [ ] Documentation covers the new dataset thoroughly.

Thanks for contributing
