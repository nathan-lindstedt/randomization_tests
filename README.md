# randomization_tests
THE SIGNIFICANCE OF RANDOMIZATION TESTS FOR PROGRAM ASSESSMENT WITH OBSERVATIONAL DATA: ADDRESSING THE ISSUE OF STATISTICAL INFERENCE WITH NON-PROBABILITY SAMPLES

In the book *Randomization Tests*, Edgington (1980) opens with a trenchant critique of the twinned myth of experimental design and statistical inference:   

>Experimental design books and others on the application of statistical tests to experimental data perpetuate the long-standing fiction of random sampling in experimental research. Statistical inferences are said to require random sampling and to concern population parameters. In experimentation, however, random sampling is very infrequent; consequently, statistical inferences about populations are usually irrelevant. Thus there is no logical connection between the random sampling model and its application to data from the typical experiment. The artificiality of the random sampling assumption has undoubtedly contributed to the skepticism of some experimenters regarding the value of statistical tests. What is a more important consequence of failure to recognize the prevalence of nonrandom sampling in experimentation, however, is overlooking the need for special statistical procedures that are appropriate for nonrandom samples. As a result, the development and application of randomization tests have suffered.

>Randomization tests are statistical tests in which the data are repeatedly divided, a test statistic (e.g., t or F) is computed for each data division, and the proportion of the data divisions with as large a test statistic value an the value for the obtained results determines the significance of the results. For testing hypotheses about experimental treatment effects, random assignment but not random sampling is required. In the absence of random sampling the statistical inferences are restricted to the subjects actually used in the experiment, and generalization to other subjects must be justified by non-statistical argument.

>Random assignment is the only random element necessary for determining the significance of experimental results by the randomization test procedure; therefore assumptions regarding random sampling and those regarding normality, homogeneity of variance, and other characteristics of randomly sampled populations, are unnecessary. Thus, any statistical test, no matter how simple or complex, is transformed into a distribution-free test when significance is determined by the randomization test procedure. For any experiment with random assignment, the experimenter can guarantee the validity of any test [they want] to use by determining significance by the randomization test procedure. Chapter 1 summarizes various advantages of the randomization test procedure, including its potential for developing statistical tests to meet the special requirements of a particular experiment, and its usefulness in providing for the valid use of statistical tests on experimental data from a single subject.

>A great deal of computation is involved in performing a randomization test and, for that reason, such a means of determining significance was impractical until recent years, when computers became accessible to experimenters. As the use of computers is essential for the practical application of randomization tests, computer programs for randomization tests accompany discussions throughout the book. The programs will be useful for a number of practical applications of randomization tests, but their main purpose is to show how programs for randomization tests are written.

>Inasmuch as the determination of significance by the randomization test procedure makes any of the hundreds (perhaps thousands) of published statistical tests into randomization tests, the discussion of application of randomization tests in this book cannot be exhaustive. Applications in the book have been selected to illustrate different facets of randomization tests so that the experimenter will have a good basis for generalizing to other applications. (P. v-vii)

He then continues by sketching the outline of a solution, describing the intuition behind a simple but expensive test that leverages the notions of permutation and random assignment to address the issue of non-probability (or non-random) samples:

>A randomization test is a permutation test based on randomization (random assignment), where the test is carried out in the following manner. A test statistic is computed for the experimental data, then the data are permuted (divided or rearranged) repeatedly in a manner consistent with the random assignment procedure, and the test statistic is computed for each of the resulting data permutations. These data permutations, including the one representing the obtained results, constitute the reference set for determining significance. The proportion of data permutations in the reference set that have test statistic values greater than or equal to (or, for certain test statistics, less than or equal to) the value for the experimentally obtained results is the P-value (significance or probability value). If, for example, the proportion is 0.02, the P-value is 0.02, and the results are significant at the 0.05 but not the 0.01 level of significance. Determining significance on the basis of a distribution of test statistics generated by permuting the data is characteristic of all permutation tests; it is when the basis for permuting the data is random assignment that a permutation test is called a randomization test. (P. 1)

Given the language of "experimentation" used throughout these passages, it is perhaps unsurprising that the application of randomization tests or permutation tests to experimental data is more familiar to researchers within the behavioral sciences (e.g., Mewhort, Johns, and Kelly [2010](https://link.springer.com/article/10.3758/BRM.42.2.366)) and the medical sciences (e.g., Rigdon and Hudgens [2014](https://onlinelibrary.wiley.com/doi/pdf/10.1002/sim.6384?casa_token=hpwlySMrlmcAAAAA:7DOYCE4Z4XD6leNc2Z5hplMK3JjuLgn7JAkiWzm0EpXd2CLUPxJYn_1RJ7cLv0DG9vcyFK0ztSuXkuCV)) as a corrective for their frequently less than ideal sampling conditions. On face value, lesser known is its relevance for observational data within the social sciences (yet see Taylor [2020](https://journals.sagepub.com/doi/pdf/10.1177/1536867X20930999)). However, it should be noted that there is an established history of randomization tests and permutation tests within social network analysis as it is employed by the QAP (Hubert and Schultz [1976](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/j.2044-8317.1976.tb00714.x)) and MRQAP (Krackhardt [1988](https://pdf.sciencedirectassets.com/271850/1-s2.0-S0378873300X00688/1-s2.0-0378873388900044/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJL%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAKzTmXIkhJsb5EuJvxODZLN8JowCSChR%2Fv9zsI3FJuYAiEAm5%2Ffg4XTRg1nYJW4lEeeG3dTqj4vBDbYcP525UXAyTMqswUIOxAFGgwwNTkwMDM1NDY4NjUiDHXujRW71QeCiOHpzyqQBZfLVS8JnFficOqST4t9Xdd%2Fmxrny%2BW7MP96pP2pDKdkAckKtbsIce1B8OSbpXJWx8tYtB6GiUJIpH045q5dIhLcCFbrzhdSENXSsBHs1fxabNSeeCggUUG4QD%2F7E7Bl7S3Tje8Ff0sSYz6Dd2cezfshtmewuMe2GufDzy%2FSNuRHdKU6rXAoR00M%2BAAAX7c%2FW%2FX8ldaEr8PQ8F1NEDGh0q9TN85HZGSkIcYkTtwTPh0ES%2Be3s%2Fm%2FxhPpn7mFmAUMet8VXdDd3xkaR8eqJf7uBn5hs%2BGGpvah4J0MoT6hQGjv0retK7TJO0aRZtIqQh9O%2F0%2FcJFSLlk1Cej3Kgrf5K0FiZMzWEjhGsrj0F9xIVtSp9buuyI4%2FGSzEOxKShaMxdUzaYm5eAePb2HdaBCGbdvrSy4S%2F9Hyknr06DMX237iRsJ8H74%2B30%2BxdNyuJY8I8d0u9Jal49BgSe%2BKwCDa%2FaZBrCgDQH1BzXaXL0cgEQdDSOYksW8iSS%2FPC5hBdge0sGa0hJKupznZP4nU%2FQRIjdRRZRdzVLwrObCxQroWtLYWXAVD0nlTrq8S3mZfTEgvR9yvfumLwE1TbGqrq3DjzQotT1924YfJwlXCnQ48Lpx5oq7ZdLDoWVLOydRHfI0Gj9Q8Fr0sYTvob4Gfz0lDEensU2kd1zTztDQl1XQ0c9gxGDm5tafH8SM44zlogSdXYVcMP2mg%2FHQ%2B%2FpXP%2BDwnL8Z7C1Xx0wtnSpQ6x2pmqR%2F%2FNzKgBJXgDBWODG9sz9%2Fh44s%2BrditcZIhkd9SpAHIOlvSERP8Ce7%2F2%2BGCDyDmCNhOLrl%2FKUAu8%2BCq9g82T9hvv6cQyqpCaixCB32jXaPCQ3Xr4%2FAw1hAwwYSbQOsaleWF2MMabzrMGOrEBytI66VrFfqk4ufolkrgzoEanpv0Yt6zqRC9JDJUzBlyCJSLh4fCtA3gldnUv2v2rS06rGGNKlGviG5oe7zeeYXj5JZI9T43mAV%2FswYHDecgHB7DTnrN8Ue%2FojkAW5yORuvlg9YY3SDtUnpcIHBBr1Fdj5PEU%2BH%2BTyTTQ%2BBaOt0O%2Fz72vPIHmtBYP11bO9EiG2aIzxYfco87pl%2Fa39GHG2P1qIpbXIpeQYgMOFKtfOzQR&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240620T032847Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYXFLBSTUL%2F20240620%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9ab94a765bf0fad184cdd35ec72f63bd960901c3a3a5ced14b507e55262f44bf&hash=4a41aed56c6a4084c03831b5c69fdc97c82492cd7bc4794362d4b4eba7e55bf8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0378873388900044&tid=spdf-aa7718c4-c68b-4b03-9128-45685c3c1223&sid=9a6b520675618341271882b98a70d1b4c58fgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=13135f5e565707030252&rr=8968abb5de6ba332&cc=us)) techniques. That being said, the idea that randomization tests or permutation tests can be applied to observational data and not just experimental data is well founded (see Box and Andersen [1953](https://repository.lib.ncsu.edu/server/api/core/bitstreams/804fd57d-c6a3-48c2-925d-cbbd4b7e0a5b/content); Chung and Fraser [1958](https://www.jstor.org/stable/2282050); Rubin [1974](https://psycnet.apa.org/record/1975-06502-001)).

Part of the reason for its unfamiliarity in the context of observational data are the limitations of such tests to within-sample conclusions. Fortunately, given the scope of research questions on program assessment under non-experimental settings, which lack the need to generalize to a larger hypothetical population to answer, these limitations do not place any greater constraints on their scope than is needed for an answer. For example, if an analyst desires to assess whether a program successfully resulted in some significantly changed outcome using a non-probability sample, all that analyst cares about is that outcome for that sample and if it was statistically significant. There is no need to generalize to a larger hypothetical population to complete that assessment. Furthermore, analyst concerns regarding the dread of "self-selection" in experimental designs can be assuaged by reframing the understanding of the kind of hypotheses tested in terms of the more limited observational assessments, where data is collected by recording events as they "naturally" take place with no manipulations. As program participants are no longer subject to intervention group and control group assignments, but are instead observed to take part in some behavior, the analyst can only offer evidence that those engaging in that behavior differed significantly from their counterparts through the assumption of exchangability. That is, the assumption that the null hypothesis of no difference should hold for those in the sample when there actually is no difference. Here random assignment being induced via the computational procedure utilized.

Enter the randomization test: a non-parametric method free from distributional assumptions. A succinct overview of permutation methods is given in an article by Berry, Johnston, and Mielke ([2011](https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wics.177?casa_token=U1vt9mhFqrgAAAAA:406S-KwWKy3A6VvTqlKf3zq5JNUaK_401ousxa1HgrZ8kbfPQMN-cku3PH-0gx5JoE2ZYZfIztC_Mfxc)).

References 

Berry, K. J., Johnston, J. E., and P. W. Mielke. 2011. "Permutation methods." *Wiley Interdisciplinary Reviews: Computational Statistics*, 3(6):527-542.

Box, G. E. and S. L. Andersen. 1954. “Robust tests for variances and effect of non-normality and variance heterogeneity on standard tests.” *Technical Report, North Carolina State University Institute of Statistics Mimeo Series.*

Chung, J. H. and D. A. S. Fraser. 1958. "Randomization tests for a multivariate two-sample problem." *Journal of the American Statistical Association*, 53(283):729–735.

Edgington, E. S. 1980. *Randomization tests.* 2nd Ed. New York, NY: Marcel Dekker, Inc.

Hubert, L. J. and J. Schultz. 1976. "Quadratic assignment as a general data analysis strategy." *British Journal of Mathematical and Statistical Psychology*, 29:190-241.

Krackhardt, D. 1988. "Predicting with networks: nonparametric multiple regression analysis of dyadic data." *Social Networks*, 10:359–381.

Mewhort, D. J. K., Johns, B. T., and M. A. Kelly. 2010. "Applying the permutation test to factorial designs." *Behavior Research Methods*, 42:366–372.

Rigdon, J. and M. G. Hudgens. 2015. "Randomization inference for treatment effects on a binary outcome." *Statistics in Medicine*, 34(6):924-935.

Rubin, D. B. 1974. "Estimating causal effects of treatments in randomized and nonrandomized studies." *Journal of educational Psychology*, 66:688–701. 

Taylor, M. A. 2020. "Visualization strategies for regression estimates with randomization inference." *The Stata Journal*, 20(2):309-335.
