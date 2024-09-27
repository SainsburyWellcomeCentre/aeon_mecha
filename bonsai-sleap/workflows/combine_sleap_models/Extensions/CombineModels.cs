using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;
using Accord.Math.Optimization;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CombineModels
{
    static double EucDist(Point2f point, Point2f other)
    {
        var dx = point.X - other.X;
        var dy = point.Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    public IObservable<PoseIdentityCollection> Process(IObservable<Tuple<PoseIdentityCollection, PoseCollection>> source)
    {
        return source.Select(value =>
        {
            var poseIdentities = value.Item1;
            var poses = value.Item2;
            var output = new PoseIdentityCollection(poses.Image, poses.Model);

            // Return empty ouput if no poses or identities
            if (poses.Count == 0 || poseIdentities.Count == 0)
            {
                Console.WriteLine("No poses or no identities detected for this frame, returning empty output");
                return output;
            } 

            // Check for and remove identities with NaN centroids in poseIdentities
            List<string> assignedClasses = new List<string>();
            for (int i = 0; i < poseIdentities.Count; i++)
            {
                if (double.IsNaN(poseIdentities[i].Centroid.Position.X))
                {
                    poseIdentities.RemoveAt(i);
                    i--;
                }
                else
                {
                    assignedClasses.Add(poseIdentities[i].Identity);
                }
            }
            
            // Initialise a list as long as poses.count to the key that will be used to do the distance calculations with 'spine2' as default
            // Check each pose and if spine2 is NaN, try to use spine1 or spine3 and update the key in the list
            // If all spine1, spine2, and spine3 are NaN, and there is >1 pose for which this is the case (i.e., you can't infer the ID to be the only remaining one), remove them from poses
            int nanSpineCount = 0;
            foreach (var pose in poses)
            {
                if (double.IsNaN(pose["spine1"].Position.X) && double.IsNaN(pose["spine2"].Position.X) && double.IsNaN(pose["spine3"].Position.X))
                {
                    nanSpineCount++;
                }
            }
            List<string> keyList = new List<string>();
            for (int i = 0; i < poses.Count; i++)
            {
                keyList.Add("spine2");
                if (double.IsNaN(poses[i]["spine2"].Position.X))
                {
                    Console.WriteLine("Spine2 is NaN, trying to use spine1 or spine3");
                    if (!double.IsNaN(poses[i]["spine1"].Position.X))
                    {
                        keyList[i] = "spine1";
                        Console.WriteLine("Using spine1 for distance calculation");
                    }
                    else if (!double.IsNaN(poses[i]["spine3"].Position.X))
                    {
                        keyList[i] = "spine3";
                        Console.WriteLine("Using spine3 for distance calculation"); 
                    }
                    else if (nanSpineCount > 1)
                    {
                        poses.RemoveAt(i);
                        keyList.RemoveAt(i);
                        i--;
                        Console.WriteLine("All spine1, spine2, and spine3 are NaN for >1 of the poses, removing pose");
                    }
                }
            }

            // STEP 1: Calculate Distance Matrix
            double[][] distanceMatrix = new double[poses.Count][];
            for (int i = 0; i < poses.Count; i++)
            {
                distanceMatrix[i] = new double[poseIdentities.Count]; // Initialize inner array
                var poseCentroid = poses[i][keyList[i]].Position;
                for (int j = 0; j < poseIdentities.Count; j++)
                {
                    distanceMatrix[i][j] = EucDist(poseCentroid, poseIdentities[j].Centroid.Position);
                    // If spine1, spine2, and spine3 of the pose are all NaN just set the distance to max value
                    if (double.IsNaN(distanceMatrix[i][j])) 
                    {
                        Console.WriteLine("NaN distance detected, setting to max value");
                        distanceMatrix[i][j] = 10000;
                    }
                }
            }

            // Check if distance matrix only contains 10000s, if so return empty output
            if (distanceMatrix.All(x => x.All(y => y == 10000)))
            {
                Console.WriteLine("Distance matrix not valid, returning empty output");
                return output;
            }

            // STEP 2: Apply Munkres Algorithm
            var munkres = new Munkres(distanceMatrix);
            bool success = munkres.Minimize();
            if (success == false) 
            {
                Console.WriteLine("Munkres failed, returning empty output");
                return output; // Return empty output if Munkres fails
            }
            double[] assignments = munkres.Solution;
            
            // STEP 3: Update Pose Identities based on assignments
            IEnumerable<string> allClasses = poseIdentities.Model.ClassNames;
            List<string> missingClasses = allClasses.Except(assignedClasses).ToList();
            for (int i = 0; i < assignments.Length; i++)
            {
                bool inferID = false;
                int assignment = (int)assignments[i];
                var pose = poses[i];
                if (double.IsNaN(assignments[i]))
                {
                    // If we have all poses but one identity less than expected, we can infer the missing ID
                    if (poses.Count == poseIdentities.Model.ClassNames.Count && missingClasses.Count == 1)
                    {
                        Console.WriteLine("1. ID of mouse at pos: " + pose.Centroid.Position + " inferred to be: " + missingClasses[0]);
                        inferID = true;
                    }
                    // Skip otherwise
                    else continue;
                }
                // Check distance threshold (approx 5cm) in case a pose is assigned to an identity that is not close to it (can happen if not all mice are detected)
                else if (distanceMatrix[i][assignment] > 25 && distanceMatrix[i][assignment] != 10000) 
                {
                    // If there is one missing pose and one missing identity, and there is a large distance between a pose p and assigned identity,
                    // we can assume that the assigned identity likely belongs to the missing pose, and infer the correct identity for the current pose p
                    if (poses.Count == poseIdentities.Model.ClassNames.Count - 1 && missingClasses.Count == 1)
                    {
                        Console.WriteLine("2. ID of mouse at pos: " + pose.Centroid.Position + " inferred to be: " + missingClasses[0]);
                        Console.WriteLine("Distance matrix for reference:");
                        for (int j = 0; j < distanceMatrix.Length; j++)
                        {
                            Console.WriteLine(string.Join(", ", distanceMatrix[j]));
                        }
                        inferID = true;
                    }
                    // Skip otherwise
                    else continue;
                }
                var updatedPoseIdentity = new PoseIdentity(pose.Image, pose.Model);
                updatedPoseIdentity.Centroid = pose.Centroid;
                foreach (var bodypart in pose)
                {
                    updatedPoseIdentity.Add(bodypart);
                }
                if (inferID)
                {
                    updatedPoseIdentity.Identity = missingClasses[0];
                    updatedPoseIdentity.IdentityIndex = allClasses.ToList().IndexOf(missingClasses[0]);
                    updatedPoseIdentity.Confidence = float.NaN;
                }
                else
                {
                    var identity = poseIdentities[assignment];
                    updatedPoseIdentity.Identity = identity.Identity;
                    updatedPoseIdentity.IdentityIndex = identity.IdentityIndex;
                    updatedPoseIdentity.Confidence = identity.Confidence;
                }
                output.Add(updatedPoseIdentity);
            }
            return output;
        });
    }
}

// POSE ONLY
// public class CombineModels
// {
//     public IObservable<PoseCollection> Process(IObservable<Tuple<PoseIdentityCollection, PoseCollection>> source)
//     {
//         return source.Select(value =>
//         {
//             var poseIdentities = value.Item1;
//             var poses = value.Item2;
//             var output = new PoseCollection(poses.Image, poses.Model);

//             if (poses.Count == 0) return output; // Return empty ouput if no poses 

//             for (int i = 0; i < poses.Count; i++)
//             {
//                 if (double.IsNaN(poses[i].Centroid.Position.X) || double.IsNaN(poses[i].Centroid.Position.Y))
//                 {
//                     Console.WriteLine("Pose centroid is NaN");
//                     Console.WriteLine("Spine1: " + poses[i]["spine1"].Position);
//                     Console.WriteLine("Spine3: " + poses[i]["spine3"].Position);
//                 }
//                 output.Add(poses[i]);
//             }
//             return output;
//         });
//     }
// }

// IDENTITY ONLY
// public class CombineModels
// {
//     public IObservable<PoseIdentityCollection> Process(IObservable<Tuple<PoseIdentityCollection, PoseCollection>> source)
//     {
//         return source.Select(value =>
//         {
//             var poseIdentities = value.Item1;
//             var poses = value.Item2;
//             var output = new PoseIdentityCollection(poses.Image, poses.Model);

//             if (poseIdentities.Count == 0) return output; // Return empty output if no identities

//             for (int i = 0; i < poseIdentities.Count; i++)
//             {
//                 if (double.IsNaN(poseIdentities[i].Centroid.Position.X) || double.IsNaN(poseIdentities[i].Centroid.Position.Y))
//                 {
//                     Console.WriteLine("Identity " + i + " centroid is NaN");
//                 }
//                 output.Add(poseIdentities[i]);
//             }
//             return output;
//         });
//     }
// }