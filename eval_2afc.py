def dreamsim_eval(self):
    data_loader, dataset_size = NightDataset(config=self.config, batch_size=self.config.batch_size,
                                             split='test_imagenet').get_dataloader()
    no_imagenet_data_loader, no_imagenet_dataset_size = NightDataset(config=self.config,
                                                                     batch_size=self.config.batch_size,
                                                                     split='test_no_imagenet').get_dataloader()
    print(len(data_loader), len(no_imagenet_data_loader))
    imagenet_score = self.get_2afc_score_eval(data_loader)
    logging.info(f"ImageNet 2AFC score: {str(imagenet_score)}")
    torch.cuda.empty_cache()
    no_imagenet_score = self.get_2afc_score_eval(no_imagenet_data_loader)
    logging.info(f"No ImageNet 2AFC score: {str(no_imagenet_score)}")
    overall_score = (imagenet_score * dataset_size + no_imagenet_score * no_imagenet_dataset_size) / (
            dataset_size + no_imagenet_dataset_size)
    logging.info(f"Overall 2AFC score: {str(overall_score)}")


def get_2afc_score_eval(self, test_loader):
    logging.info("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []
    # with torch.no_grad()
    for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img_ref, img_left, img_right, target = img_ref.cuda(), img_left.cuda(), \
            img_right.cuda(), target.cuda()
        dist_0, dist_1, target = self.one_step_2afc_score_eval(img_ref, img_left, img_right, target)
        d0s.append(dist_0)
        d1s.append(dist_1)
        targets.append(target)

    twoafc_score = get_2afc_score(d0s, d1s, targets)
    return twoafc_score