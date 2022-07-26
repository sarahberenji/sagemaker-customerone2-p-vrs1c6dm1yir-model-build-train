def get_latest_model_metadata(sm_client,model_package_group_name):
    approved_packages = []
    for p in sm_client.get_paginator('list_model_packages').paginate(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='PendingManualApproval',
            SortBy="CreationTime",
            SortOrder="Descending",
    ):
        approved_packages.extend(p["ModelPackageSummaryList"])
    model_metadata = sm_client.describe_model_package(ModelPackageName =approved_packages[0]["ModelPackageArn"])
    print(model_metadata)
    return model_metadata