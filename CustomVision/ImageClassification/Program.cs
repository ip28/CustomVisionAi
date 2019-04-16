using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace ImageClassification
{
    class Program
    {
        private const string CognitiveApiEndpoint = "https://australiaeast.api.cognitive.microsoft.com";
        private const string ProjectName = "Bushfire Identification Project";
        private const string CompactDomainName = "General (compact)";
        private const string DockerFileName = "TrainingDockerFile";
        private const string ModelName = "bushfireModel";
        private const string IterationPlatform = "DockerFile";
        private const string SubscriptionResourceId = "<Resource ID for Prediction>";
        private const string BaseImageFolderPath = "Images\\Bushfire";
        private const string TestImagePath = "Test\\Bugaboo_forest_fire.jpg";
        private const string DockerFlavour = "Linux";
        private static Dictionary<string, List<string>> ImagesByFolder = new Dictionary<string, List<string>>();
        private static MemoryStream testImage;

        private static bool IsNewProject = false;
        private static Iteration Iteration;
        static async Task  Main(string[] args)
        {
            // Add your training & prediction key from the settings page of the portal
            string trainingKey = "<training key>";
            string predictionKey = "<prediction key>";

            // Create the Api, passing in the training key
            CustomVisionTrainingClient trainingApi = new CustomVisionTrainingClient()
            {
                ApiKey = trainingKey,
                Endpoint = CognitiveApiEndpoint
            };
            Project project = await GetOrCreateProject(trainingApi);
            if (IsNewProject)
            {
                Iteration = await TagTrainAndPublish(predictionKey, trainingApi, project);
            }
            Console.WriteLine("Initiating iteration export");
            var downloadUri = ExportDockerFile(trainingApi, project.Id, Iteration.Id);
            using(var client = new WebClient())
            {
                Console.WriteLine("Downloading iteration....");
                await client.DownloadFileTaskAsync(new Uri(downloadUri), $"{DockerFileName}.zip");
            }
            Console.WriteLine("\tExtracting exported iteration..");
            ZipFile.ExtractToDirectory($"{DockerFileName}.zip", DockerFileName);
            Console.WriteLine("Press any key to exit!");
            Console.ReadKey();
        }

        private static async Task<Project> GetOrCreateProject(CustomVisionTrainingClient trainingApi)
        {
            IsNewProject = false;
            var allProjects = await trainingApi.GetProjectsAsync();
            var myProject = allProjects.FirstOrDefault(p => p.Name == ProjectName);
            if (myProject != null)
            {
                var iterations = await trainingApi.GetIterationsAsync(myProject.Id);
                Iteration = iterations.FirstOrDefault();
                return myProject;
            }
            return await CreateNewProject(trainingApi);
        }
        private static async Task<Project> CreateNewProject(CustomVisionTrainingClient trainingApi)
        {
            IsNewProject = true;
            var allDomains = await trainingApi.GetDomainsAsync();
            var compactProject = allDomains.FirstOrDefault(d => d.Name == CompactDomainName);
            // Create a new project
            Console.WriteLine("Creating new project:");
            var project = await trainingApi.CreateProjectAsync(ProjectName, domainId: compactProject.Id);
            return project;
        }


        private static async Task<Iteration> TagTrainAndPublish(string predictionKey, CustomVisionTrainingClient trainingApi, Project project)
        {
            Console.WriteLine("\tUploading images");
            LoadImagesFromDisk();

            await TagAndUploadImages(trainingApi, project);

            // Now there are images with tags start training the project
            Console.WriteLine("\tTraining");
            Iteration iteration = Train(trainingApi, project);

            // The iteration is now trained. Publish it to the prediction end point.
            string publishedModelName = await PublishIteration(trainingApi, project, iteration);
            Console.WriteLine("Done!\n");
            // Now there is a trained endpoint, it can be used to make a prediction

            // Create a prediction endpoint, passing in obtained prediction key
            await CreatePredictionEndpoint(predictionKey, project, publishedModelName);
            return iteration;
        }

        private static void LoadImagesFromDisk()
        {
            // this loads the images to be uploaded from disk into memory
            var directoryList = Directory.GetDirectories(BaseImageFolderPath);
            foreach (var dirPath in directoryList)
            {
                var currentFolderName = Path.GetFileName(dirPath);
                if (!string.Equals(currentFolderName, "Test", StringComparison.OrdinalIgnoreCase))
                {
                    var imagePaths = Directory.GetFiles(dirPath).ToList();
                    ImagesByFolder.Add(currentFolderName.ToLower(), imagePaths);
                }

            }
            testImage = new MemoryStream(File.ReadAllBytes(Path.Combine(BaseImageFolderPath, TestImagePath)));
        }

        private static Iteration Train(CustomVisionTrainingClient trainingApi, Project project)
        {
            var iteration = trainingApi.TrainProject(project.Id);

            // The returned iteration will be in progress, and can be queried periodically to see when it has completed
            while (iteration.Status == "Training")
            {
                Thread.Sleep(1000);

                // Re-query the iteration to get it's updated status
                iteration = trainingApi.GetIteration(project.Id, iteration.Id);
            }

            return iteration;
        }

        private static async Task TagAndUploadImages(CustomVisionTrainingClient trainingApi, Project project)
        {
            var createdTags = new List<Tag>();
            //uploaded in a single batch 
            foreach (var taggedPair in ImagesByFolder)
            {
                var imageFilePaths = taggedPair.Value;
                var splittedTags = taggedPair.Key.Split('-');
                var imageIagIds = new List<Guid>();
                foreach (var t in splittedTags)
                {
                    Tag tag;
                    var existingTag = createdTags.FirstOrDefault(tg => tg.Name == t);
                    if (existingTag != null)
                    {
                        tag = existingTag;
                    }
                    else
                    {
                        tag = await trainingApi.CreateTagAsync(project.Id, t);
                        createdTags.Add(tag);
                    }
                    imageIagIds.Add(tag.Id);

                }
                var imageFiles = imageFilePaths.Select(img => new ImageFileCreateEntry(Path.GetFileName(img), File.ReadAllBytes(img))).ToList();
                await trainingApi.CreateImagesFromFilesAsync(project.Id, new ImageFileCreateBatch(imageFiles, imageIagIds));
            }
        }

        private static async Task<string> PublishIteration(CustomVisionTrainingClient trainingApi, Project project, Iteration iteration)
        {
            var publishedModelName = ModelName;
            var predictionResourceId = SubscriptionResourceId;
            await trainingApi.PublishIterationAsync(project.Id, iteration.Id, publishedModelName, predictionResourceId);
            return publishedModelName;
        }
        private static async Task CreatePredictionEndpoint(string predictionKey, Project project, string publishedModelName)
        {
            CustomVisionPredictionClient endpoint = new CustomVisionPredictionClient()
            {
                ApiKey = predictionKey,
                Endpoint = CognitiveApiEndpoint
            };

            // Make a prediction against the new project
            Console.WriteLine("Making a prediction:");
            var result = await endpoint.ClassifyImageAsync(project.Id, publishedModelName, testImage);

            // Loop over each prediction and write out the results
            foreach (var c in result.Predictions)
            {
                Console.WriteLine($"\t{c.TagName}: {c.Probability:P1}");
            }
        }


        private static string ExportDockerFile(CustomVisionTrainingClient trainingApi, Guid projectId, Guid iterationId)
        {
            var allExports = trainingApi.GetExports(projectId, iterationId);
            if (allExports.Any())
            {
                while (string.IsNullOrWhiteSpace(allExports.FirstOrDefault().DownloadUri))
                {
                    allExports = trainingApi.GetExports(projectId, iterationId);
                }
                return allExports.FirstOrDefault().DownloadUri;
            }
            else
            {
               trainingApi.ExportIteration(projectId, iterationId, IterationPlatform, DockerFlavour);
            }
            return ExportDockerFile(trainingApi, projectId, iterationId);
        }

    }
}
