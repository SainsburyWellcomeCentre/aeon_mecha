/* 
Greedy algo: 
    - Initially assign all poses in `PoseCollection` an ID of "Unknown".
    - Take centroid from `PoseCollection`, find closest centroid in `PoseIdentityCollection`; assign this ID to pose.
    - Repeat until we run out of poses or run out of IDs.
Output in `PoseIdentityCollection` format.
*/

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
public class CombineModelsGreedy
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
            foreach (var pose in poses)
            {
                var minDistance = double.MaxValue;
                var updatedPoseIdentity = new PoseIdentity(pose.Image, pose.Model);
                updatedPoseIdentity.Centroid = pose.Centroid;
                foreach (var bodypart in pose)
                {
                    updatedPoseIdentity.Add(bodypart);
                }

                foreach (var identity in poseIdentities)
                {
                    var distance = EucDist(pose.Centroid.Position, identity.Centroid.Position);
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        updatedPoseIdentity.Identity = identity.Identity;
                        updatedPoseIdentity.IdentityIndex = identity.IdentityIndex;
                        updatedPoseIdentity.Confidence = identity.Confidence;
                    }
                }
                // Console.WriteLine("ID: " + updatedPoseIdentity.Identity + "; Pos: " + pose.Centroid.Position);
                output.Add(updatedPoseIdentity);
            }
            return output;
        });
    }
}
