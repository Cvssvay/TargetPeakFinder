<<<<<<< HEAD
# TargetedPeakFinder

# Managing Large Datasets in Git Repositories with Git LFS

When working on machine learning or data science projects, you often have large datasets that you don't want to include directly in your Git repository. Instead, you can use Git LFS (Large File Storage) to handle these large files efficiently. This tutorial will guide you through the process of setting up `.gitignore` and Git LFS to manage large datasets.

## Prerequisites

- Basic knowledge of Git.
- Git and Git LFS installed on your machine.

## Step 1: Create/Modify `.gitignore`

The `.gitignore` file tells Git which files (or patterns) it should ignore. If you don't already have a `.gitignore` file in the root of your repository, create one. To ignore all JSON files in the `data/annotations/` directory and its subdirectories, add the following lines to your `.gitignore` file:

```gitignore
# Ignore all JSON files in data/annotations and its subdirectories
data/annotations/**/*.json

# Optionally, ignore the entire data directory
# data/

## Step 2: Install Git LFS

```git lfs install```

## Step 3: Track JSON Files with Git LFS
```git lfs track "data/annotations/**/*.json"```


## Step 4: Verify and Commit Changes
The git lfs track command should have added an entry to your .gitattributes file. Ensure it includes the following:



```data/annotations/**/*.json filter=lfs diff=lfs merge=lfs -text```


=======
# TargetPeakFinder
>>>>>>> 119c59e4462890dca7299a5c5168155da20686ef
