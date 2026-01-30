# Results

- Accuracy
	- Augmentation
		- Normal Model: 84.06% -> 79.70%
		- Contrastive Model: 82.95% -> 79.24%
		- Adversarial Model: 75.17% -> 72.54%
		- Adversarial Contrastive: 51.79% -> 51.28%
		- Certified: 52.07% -> 50.51%
		- Certified Contrastive: 33.69% -> 33.95%
	- No Augmentation
		- Normal Model: 98.84% -> 67.71%
		- Contrastive Model: 99.05% -> 67.53%
		- Adversarial Model: 88.78% -> 68.63%
		- Adversarial Contrastive: 79.79% -> 67.78%
		- Certified: 81.59% -> 54.21%
		- Certified Contrastive: 13.33% -> 13.39%

---

> ASR = Attack Success Rate
> Eps: 1/255
	> Normal Model
		- Test ASR: 16.233848571777344%;
	> Contrastive Model
		- Test ASR: 14.893348693847656%;
	> Adversarial Model
		- Test ASR: 4.923459053039551%;
	> Adversarial Contrastive
		- Test ASR: 5.269320964813232%;
	> Certified
		- Test ASR: 13.787638664245605%;
	> Certified Contrastive
		- Test ASR: 18.79787826538086%;
	> Normal Model
		- Test ASR: 34.61822509765625%;
	> Contrastive Model
		- Test ASR: 43.625057220458984%;
	> Adversarial Model
		- Test ASR: 5.144272804260254%;
	> Adversarial Contrastive
		- Test ASR: 5.283352851867676%;
	> Certified
		- Test ASR: 16.44845962524414%;
	> Certified Contrastive
		- Test ASR: 5.983545303344727%;
> Eps: 2/255
	> Normal Model
		- Test ASR: 33.672061920166016%;
	> Contrastive Model
		- Test ASR: 31.3012752532959%;
	> Adversarial Model
		- Test ASR: 9.557302474975586%;
	> Adversarial Contrastive
		- Test ASR: 10.050741195678711%;
	> Certified
		- Test ASR: 26.465927124023438%;
	> Certified Contrastive
		- Test ASR: 35.71007537841797%;
	> Normal Model
		- Test ASR: 64.33318328857422%;
	> Contrastive Model
		- Test ASR: 72.41226196289062%;
	> Adversarial Model
		- Test ASR: 10.886038780212402%;
	> Adversarial Contrastive
		- Test ASR: 11.009445190429688%;
	> Certified
		- Test ASR: 31.71676254272461%;
	> Certified Contrastive
		- Test ASR: 10.99476432800293%;
> Eps: 4/255
	> Normal Model
		- Test ASR: 63.59302520751953%;
	> Contrastive Model
		- Test ASR: 61.17632293701172%;
	> Adversarial Model
		- Test ASR: 19.12839698791504%;
	> Adversarial Contrastive
		- Test ASR: 20.023418426513672%;
	> Certified
		- Test ASR: 48.256736755371094%;
	> Certified Contrastive
		- Test ASR: 59.605186462402344%;
	> Normal Model
		- Test ASR: 90.03101348876953%;
	> Contrastive Model
		- Test ASR: 94.77268981933594%;
	> Adversarial Model
		- Test ASR: 22.602739334106445%;
	> Adversarial Contrastive
		- Test ASR: 22.919126510620117%;
	> Certified
		- Test ASR: 55.70717239379883%;
	> Certified Contrastive
		- Test ASR: 17.576663970947266%;
> Eps: 8/255
	> Normal Model
		- Test ASR: 93.31326293945312%;
	> Contrastive Model
		- Test ASR: 91.15234375%;
	> Adversarial Model
		- Test ASR: 37.63618850708008%;
	> Adversarial Contrastive
		- Test ASR: 40.47618865966797%;
	> Certified
		- Test ASR: 78.09033203125%;
	> Certified Contrastive
		- Test ASR: 85.53329467773438%;
	> Normal Model
		- Test ASR: 98.86280059814453%;
	> Contrastive Model
		- Test ASR: 99.8074951171875%;
	> Adversarial Model
		- Test ASR: 45.58437728881836%;
	> Adversarial Contrastive
		- Test ASR: 44.465763092041016%;
	> Certified
		- Test ASR: 79.42098236083984%;
	> Certified Contrastive
		- Test ASR: 28.19745635986328%;
> Eps: 16/255
	> Normal Model
		- Test ASR: 99.8996353149414%;
	> Contrastive Model
		- Test ASR: 99.24271392822266%;
	> Adversarial Model
		- Test ASR: 71.2867202758789%;
	> Adversarial Contrastive
		- Test ASR: 70.70648193359375%;
	> Certified
		- Test ASR: 97.64263153076172%;
	> Certified Contrastive
		- Test ASR: 98.70359802246094%;
	> Normal Model
		- Test ASR: 99.92615509033203%;
	> Contrastive Model
		- Test ASR: 100.0%;
	> Adversarial Model
		- Test ASR: 77.3535385131836%;
	> Adversarial Contrastive
		- Test ASR: 76.151123046875%;
	> Certified
		- Test ASR: 88.49345397949219%;
	> Certified Contrastive
		- Test ASR: 67.53926849365234%;

---

> ab-crown
{
   "Normal Model":{
      "timeout":8,
      "safe":9,
      "safe-incomplete":3,
   },
   "Contrastive Model":{
      "timeout":12,
      "safe":7,
      "safe-incomplete":1
   },
   "Adversarial Model":{
		  "timeout": 0,
			"safe": 0,
      "safe-incomplete":20,
   },
   "Adversarial Contrastive":{
			"timeout": 0,
			"safe": 0,
      "safe-incomplete":20
   },
	 "Certified": {
			"timeout": 1,
			"safe": 0,
      "safe-incomplete":19
	 },
	 "Certified Contrastive": {
			"timeout": 0,
			"safe": 0,
      "safe-incomplete":20
	 }
}